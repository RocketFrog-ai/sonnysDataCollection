"""Assemble the proforma backtest analysis notebook, then it is executed with
`jupyter nbconvert --execute`.  Kept as a script so the notebook is reproducible.
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# Proforma Backtest — do the site-selection factors & projections match reality?

**Question.** Sonny's proformas score a prospective site on ~10 site-selection factors, roll
them into a `cumulative_site_score` / target scores, and project a 5-year monthly wash-count
(`vol_monthly_y1..y5`). We now have **actual** monthly wash-count / revenue / ASP for the sites
that were actually built (address-matched to the operating panel).

This notebook asks three things:

1. **Assumption vs reality** — how far off were the projected volumes from what the site actually did?
2. **Signal** — do the factor scores (and demographics, traffic, price) actually correlate with real performance?
3. **Weightage** — the proforma weights the factors a certain way; what weighting does the *data* imply? Which factors truly move wash-count?

Data: `old-proforma-combined-monthly.csv` (1 row / proforma / month) + the per-proforma summary.
""")

code("""import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
pd.set_option('display.width', 200, 'display.max_columns', 60)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .3, 'font.size': 10})

import os as _os
BASE = '.' if _os.path.exists('old-proforma-combined-monthly.csv') else '..'
m = pd.read_csv(f'{BASE}/old-proforma-combined-monthly.csv')
s = pd.read_csv(f'{BASE}/old-proforma-combined.csv')
print('monthly rows:', len(m), '| proformas:', s.source_file.nunique(),
      '| matched:', (s.match_status != 'no_match').sum())
m.head(3)""")

md("""## 1. Build a site-level analysis table

Each proforma is one prospective site. From the monthly panel I derive a **robust "mature"
actual** per matched site: I keep only non-imputed months and take the **median of the last 12
clean months** for wash-count, revenue and blended ASP. Median-of-last-12 sidesteps both the
ramp-up and the 2020 left-censoring, and is far more stable than a full-history mean.

I also compute **actual operating-year averages** (Year 1..5 from the real open date) so the
projected `vol_monthly_yN` can be compared to the matching operating year — but only for sites
whose open date is after the panel's 2020 left-censor (earlier ones have an unknown true age).""")

code("""# --- per-site MATURE actuals (median of last 12 clean months) ---
mm = m[m.match_status.ne('no_match')].copy()
mm['imputed'] = mm['imputed'].fillna(0)
clean = mm[mm['imputed'] == 0].copy()
clean['blended_asp'] = clean['revenue'] / clean['wash_count'].replace(0, np.nan)

def mature(g):
    g = g.sort_values(['year', 'month']).tail(12)
    return pd.Series({'act_wash': g['wash_count'].median(),
                      'act_revenue': g['revenue'].median(),
                      'act_asp': g['blended_asp'].median(),
                      'act_mem_pct': g['mem_count_pct'].median(),
                      'n_clean': len(g)})

site_act = clean.groupby('source_file').apply(mature, include_groups=False).reset_index()

# --- actual operating-year monthly averages (real open dates only) ---
def open_ym(x):
    try:
        mo, yr = str(x).split('-'); return int(yr) * 12 + int(mo)
    except Exception:
        return np.nan
mm['open_idx'] = mm['match_operational_start'].map(open_ym)
mm['cal_idx'] = mm['year'] * 12 + mm['month']
mm['mo_since_open'] = mm['cal_idx'] - mm['open_idx']
mm['op_year'] = (mm['mo_since_open'] // 12) + 1
open_year = mm.groupby('source_file')['match_operational_start'].first().str[-4:].astype(float)

yr_act = (mm[(mm.imputed == 0) & mm.op_year.between(1, 5)]
          .groupby(['source_file', 'op_year'])['wash_count'].mean().unstack())
yr_act.columns = [f'act_y{int(c)}_wash' for c in yr_act.columns]
yr_act = yr_act.reset_index()
print('sites with mature actuals:', len(site_act))
site_act.describe()[['act_wash','act_revenue','act_asp','n_clean']].round(1)""")

code("""# --- assemble site-level table: proforma inputs + factor scores + actuals ---
site = s[s.match_status.ne('no_match')].drop_duplicates('source_file').copy()
site = site.merge(site_act, on='source_file', how='left').merge(yr_act, on='source_file', how='left')
site['open_year'] = site['source_file'].map(open_year)

# data-quality filter: need real clean data and a non-trivial operating site
before = len(site)
site = site[(site['n_clean'] >= 6) & (site['act_wash'] >= 50)].copy()
print(f'kept {len(site)} / {before} matched sites after quality filter '
      f'(>=6 clean months & mature wash >= 50/mo)')

FACTOR_SCORES = [c for c in site.columns if c.startswith('factor_') and c.endswith('_score')]
# drop zero-variance factors (weekly_hours is constant across all proformas -> no signal)
live_factors = [c for c in FACTOR_SCORES if site[c].std() > 1e-9]
dead = sorted(set(FACTOR_SCORES) - set(live_factors))
print('live factor scores:', len(live_factors), '| constant/no-signal:', dead)
site[['source_file','act_wash','act_revenue','act_asp','cumulative_site_score','mature_target_score','traffic_count']].head()""")

md("""## 2. Assumption vs Reality — how good were the projected volumes?

Directly compare each proforma's **projected monthly wash-count** to the **actual**. First the
mature comparison (projected mature `vol_monthly_y5` vs actual mature), then the operating-year
comparison where open dates are real.""")

code("""fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# (a) projected mature vs actual mature
d = site.dropna(subset=['vol_monthly_y5', 'act_wash'])
ax[0].scatter(d['vol_monthly_y5'], d['act_wash'], s=32, alpha=.6, edgecolor='k', lw=.4)
lim = [0, max(d['vol_monthly_y5'].max(), d['act_wash'].max()) * 1.05]
ax[0].plot(lim, lim, 'r--', lw=1, label='perfect (y=x)')
ax[0].set(xlim=lim, ylim=lim, xlabel='Projected mature monthly washes (vol_monthly_y5)',
          ylabel='Actual mature monthly washes', title='Projected vs Actual (mature)')
r = stats.pearsonr(d['vol_monthly_y5'], d['act_wash'])
ax[0].legend(title=f'n={len(d)}  r={r[0]:.2f}')

# (b) ratio distribution actual/projected
ratio = (d['act_wash'] / d['vol_monthly_y5']).replace([np.inf, -np.inf], np.nan).dropna()
ax[1].hist(ratio, bins=24, color='#4C72B0', edgecolor='w')
ax[1].axvline(1, color='r', ls='--', label='projected = actual')
ax[1].axvline(ratio.median(), color='k', ls='-', label=f'median={ratio.median():.2f}')
ax[1].set(xlabel='Actual / Projected (mature monthly washes)', ylabel='sites',
          title='Projection bias')
ax[1].legend()
plt.tight_layout(); plt.show()

mape = (np.abs(d['act_wash'] - d['vol_monthly_y5']) / d['act_wash']).median()
print(f'Median |error| vs actual (MAPE): {mape*100:.0f}%')
print(f'Median actual/projected ratio: {ratio.median():.2f}  '
      f'({\"proforma UNDER-projects\" if ratio.median()>1 else \"proforma OVER-projects\"})')
print(f'Sites where proforma over-projected (actual < projected): '
      f'{(ratio<1).mean()*100:.0f}%')""")

md("""**How to read this.** *Left:* every dot is one site — its proforma-**projected** mature
monthly washes (x-axis) vs what it **actually** did (y-axis). Dots on the red dashed line = a
perfect forecast; dots **below** the line = the proforma expected more cars than showed up.
*Right:* the actual ÷ projected ratio per site — 1.0 is spot-on. The median sits at **0.63**
and most bars are left of 1.0, so sites typically do only ~63% of the projected volume: the
proforma runs hot.""")

code("""# operating-year accuracy: projected vol_monthly_yN vs actual act_yN_wash (real opens only)
rows = []
for y in range(1, 6):
    pcol, acol = f'vol_monthly_y{y}', f'act_y{y}_wash'
    if acol not in site: continue
    d = site.dropna(subset=[pcol, acol])
    d = d[d['open_year'] >= 2021]           # only sites with a real (post-censor) open date
    if len(d) < 5: continue
    ratio = (d[acol] / d[pcol]).replace([np.inf, -np.inf], np.nan).dropna()
    rows.append({'operating_year': f'Y{y}', 'n_sites': len(d),
                 'proj_monthly_mean': d[pcol].mean(), 'actual_monthly_mean': d[acol].mean(),
                 'median_actual/proj': ratio.median(),
                 'median_abs_pct_err': (np.abs(d[acol]-d[pcol])/d[acol]).median()*100})
acc = pd.DataFrame(rows)
print('Projection accuracy by operating year (real-open sites):')
acc.round(2)""")

md("""## 3. Do the factors actually correlate with real performance?

Correlate every candidate driver — the 10 site-factor scores, the composite scores, demographics,
traffic and package prices — against the mature actual **wash-count**, **revenue** and **ASP**.
Both Pearson (linear) and Spearman (rank) so a monotone-but-nonlinear signal isn't missed.""")

code("""drivers = (live_factors +
    ['cumulative_site_score', 'cumulative_demographic_score',
     'first_year_target_score', 'second_year_target_score', 'mature_target_score',
     'traffic_count', 'demog_avg_household_size_value', 'demog_pct_pop_25_65_value',
     'demog_pct_hh_income_35k_value', 'demog_base_price_carwash_value',
     'pkg_menu1_price', 'pkg_menu3_price', 'weekly_hours_operation', 'avg_daily_wash_hours'])
drivers = [d for d in drivers if d in site.columns and site[d].std() > 1e-9]
targets = ['act_wash', 'act_revenue', 'act_asp']

def corr_table(df, drivers, target, method='pearson'):
    out = []
    for d in drivers:
        sub = df[[d, target]].dropna()
        if len(sub) < 8: continue
        if method == 'pearson':
            r, p = stats.pearsonr(sub[d], sub[target])
        else:
            r, p = stats.spearmanr(sub[d], sub[target])
        out.append({'driver': d, 'r': r, 'p': p, 'n': len(sub)})
    return pd.DataFrame(out).set_index('driver')

ct = corr_table(site, drivers, 'act_wash', 'pearson').rename(columns={'r':'pearson_r','p':'pearson_p'})
cs = corr_table(site, drivers, 'act_wash', 'spearman')[['r']].rename(columns={'r':'spearman_r'})
corr_wash = ct.join(cs).sort_values('pearson_r', key=abs, ascending=False)
print('Correlation with ACTUAL mature monthly WASH-COUNT:')
corr_wash.round(3)""")

code("""fig, ax = plt.subplots(figsize=(9, 6))
c = corr_wash.sort_values('pearson_r')
colors = ['#C44E52' if v < 0 else '#55A868' for v in c['pearson_r']]
ax.barh(c.index, c['pearson_r'], color=colors, edgecolor='k', lw=.4)
ax.scatter(c['spearman_r'], range(len(c)), color='k', s=18, zorder=3, label='Spearman (rank)')
ax.axvline(0, color='k', lw=.8)
# mark significance
for i, (idx, row) in enumerate(c.iterrows()):
    if row['pearson_p'] < 0.05:
        ax.text(row['pearson_r'] + (0.01 if row['pearson_r'] >= 0 else -0.01), i, '*',
                va='center', ha='left' if row['pearson_r'] >= 0 else 'right', fontsize=14)
ax.set(xlabel="correlation with actual mature wash-count  (bars=Pearson, dots=Spearman, * = p<0.05)",
       title='Which proforma inputs actually track real wash volume?')
ax.legend(loc='lower right'); plt.tight_layout(); plt.show()""")

md("""**How to read this.** Each bar is one proforma input's correlation with real wash volume.
**Green/right = a higher score goes with more washes; red/left = with fewer.** Longer bar =
stronger link; black dots are the rank-based (Spearman) check, and a **\\*** marks
statistically-real signal (p<0.05). Only a few inputs — led by **pay stations** and **free
vacuum slots** (both capacity/throughput) — actually track real volume; most sit near zero,
meaning they carry little real signal despite being scored.""")

code("""# categorical: does the CHOICE on key factors separate real performance?
for fac in ['factor_pay_stations_choice', 'factor_nearest_competition_choice',
            'factor_area_profile_choice', 'factor_visibility_choice']:
    g = (site.dropna(subset=['act_wash']).groupby(fac)['act_wash']
         .agg(['mean', 'median', 'count']).sort_values('median', ascending=False))
    g = g[g['count'] >= 2]
    print(f'\\n{fac}  -> actual mature monthly wash-count')
    print(g.round(0).to_string())""")

md("""## 4. Empirical weightage vs the proforma's assumed weightage

The proforma assigns each factor a slice of `cumulative_site_score`. Compare that **assumed
weight** (each factor's mean share of the cumulative score) against what the **data** implies:
the standardized regression coefficient and random-forest importance for predicting actual
wash-count. Divergence = factors the template over-/under-weights relative to reality.""")

code("""# assumed weight = each live factor's mean share of the cumulative site score
assumed = site[live_factors].abs().mean()
assumed = (assumed / assumed.sum()).rename('assumed_weight_share')

# empirical: standardized linear regression + random forest on actual wash-count
X = site[live_factors + ['traffic_count', 'demog_pct_hh_income_35k_value',
                         'demog_avg_household_size_value']].copy()
y = site['act_wash']
ok = X.notna().all(axis=1) & y.notna()
X, y = X[ok], y[ok]
Xz = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)

lin = LinearRegression().fit(Xz, y)
rf = RandomForestRegressor(n_estimators=400, random_state=0, max_depth=4).fit(Xz, y)
r2_lin = lin.score(Xz, y)

emp = pd.DataFrame({
    'std_lin_coef': pd.Series(lin.coef_, index=X.columns),
    'rf_importance': pd.Series(rf.feature_importances_, index=X.columns),
}).join(assumed)
emp['abs_corr'] = emp.index.map(lambda d: abs(site[[d,'act_wash']].dropna().corr().iloc[0,1]))
emp = emp.sort_values('rf_importance', ascending=False)
print(f'Linear R^2 (factors -> actual wash, n={len(y)}): {r2_lin:.2f}')
print(f'How well does the proforma\\'s OWN projection predict actual? '
      f"r = {site[['vol_monthly_y5','act_wash']].dropna().corr().iloc[0,1]:.2f}")
emp.round(3)""")

code("""fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
e = emp.dropna(subset=['assumed_weight_share'])
e2 = e.sort_values('assumed_weight_share')
idx = np.arange(len(e2))
ax[0].barh(idx - .2, e2['assumed_weight_share'], height=.4, label='Assumed (proforma weight share)', color='#8172B3')
ax[0].barh(idx + .2, e2['rf_importance'], height=.4, label='Empirical (RF importance)', color='#CCB974')
ax[0].set_yticks(idx); ax[0].set_yticklabels([i.replace('factor_','').replace('_score','') for i in e2.index])
ax[0].set(xlabel='weight share', title='Assumed vs Empirical factor weight')
ax[0].legend(fontsize=8)

# rank divergence scatter
ax[1].scatter(e['assumed_weight_share'].rank(), e['rf_importance'].rank(), s=40, edgecolor='k')
for i, row in e.iterrows():
    ax[1].annotate(i.replace('factor_','').replace('_score',''),
                   (row['assumed_weight_share'].__class__ and e['assumed_weight_share'].rank()[i],
                    e['rf_importance'].rank()[i]), fontsize=7, alpha=.8)
mx = len(e)
ax[1].plot([1, mx], [1, mx], 'r--', lw=1)
ax[1].set(xlabel='assumed-weight rank', ylabel='empirical-importance rank',
          title='Off-diagonal = mis-weighted factors')
plt.tight_layout(); plt.show()""")

md("""**How to read this.** *Left:* purple = how much the proforma **assumes** each factor matters
(its share of the site score); gold = how much it **actually** matters for real wash-count
(data-driven importance). Where **gold ≫ purple** the template **under-weights** that factor;
where **purple ≫ gold** it **over-weights** it. *Right:* the same as ranks — dots far off the red
line are the mis-weighted factors. The template spreads weight ~evenly, but the data says
capacity factors (pay stations, vacuums) and demographics do most of the work.""")

md("""## 5. Does the composite score predict anything?

The whole point of `cumulative_site_score` / `mature_target_score` is to rank sites. Do higher-scored
sites actually wash more cars?""")

code("""fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, col, lab in zip(ax,
        ['cumulative_site_score', 'mature_target_score', 'vol_monthly_y5'],
        ['cumulative_site_score', 'mature_target_score', 'proforma projected (vol_monthly_y5)']):
    d = site.dropna(subset=[col, 'act_wash'])
    a.scatter(d[col], d['act_wash'], s=30, alpha=.6, edgecolor='k', lw=.3)
    if len(d) > 3:
        b, m0 = np.polyfit(d[col], d['act_wash'], 1), None
        xs = np.linspace(d[col].min(), d[col].max(), 50)
        a.plot(xs, np.polyval(b, xs), 'r-', lw=1.2)
        r, p = stats.pearsonr(d[col], d['act_wash'])
        a.set_title(f'{lab}\\nr={r:.2f}, p={p:.3f}, n={len(d)}', fontsize=9)
    a.set(xlabel=lab, ylabel='actual mature monthly wash')
plt.tight_layout(); plt.show()""")

md("""**How to read this.** Does a site's *overall* proforma verdict predict reality? Each panel
plots one summary score (x) against actual mature washes (y), with a trend line and correlation
**r**. A near-flat line / small r means the score barely separates strong sites from weak ones.
All three are weak (**r ≈ 0.2**), so even the headline site score and the projected volume only
loosely track how many cars a site really washes.""")

md("""## 6. ASP & price reality check

The proforma sets menu prices; the panel reports realized ASP. Are the priced assumptions
anywhere near what customers actually pay?""")

code("""d = site.dropna(subset=['pkg_menu1_price', 'act_asp'])
d = d[(d['act_asp'] > 0) & (d['act_asp'] < 100)]   # trim ASP outliers
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].scatter(d['pkg_menu1_price'], d['act_asp'], s=30, alpha=.6, edgecolor='k', lw=.3)
lim = [0, max(d['pkg_menu1_price'].max(), d['act_asp'].max())*1.05]
ax[0].plot(lim, lim, 'r--', lw=1)
ax[0].set(xlabel='proforma base menu price ($)', ylabel='actual blended ASP ($)',
          title='Priced vs realized ASP', xlim=lim, ylim=lim)
ax[1].hist((d['act_asp'] - d['pkg_menu1_price']), bins=20, color='#4C72B0', edgecolor='w')
ax[1].axvline(0, color='r', ls='--')
ax[1].set(xlabel='actual ASP - proforma base price ($)', ylabel='sites', title='ASP gap')
plt.tight_layout(); plt.show()
print('median realized ASP:', round(d['act_asp'].median(),2),
      '| median proforma base price:', round(d['pkg_menu1_price'].median(),2))""")

md("""**How to read this.** *Left:* the proforma's base wash price (x) vs the average price
customers **actually** paid (y); the red line is 'priced = realized'. Most dots sit **above** it,
so real ASP (**$16.78** median) runs higher than the proforma's base menu price (**$10**).
*Right:* the per-site gap (actual − assumed price) — mass to the right of 0 confirms the base-price
assumption is set low.""")

md("""## 7. Findings

The cells below print the concrete numbers; the narrative summary:

- **Projections are biased and noisy.** The `Actual / Projected` ratio and MAPE quantify how far
  the proforma's `vol_monthly_yN` sat from reality — read the median ratio (>1 = the template
  systematically *under*-projects, <1 = *over*-projects) and the share of sites over-projected.
- **Only some factors carry signal.** The correlation bar chart separates factors that actually
  track real wash-count (starred = p<0.05) from the ones that are noise. `weekly_hours` is constant
  and was dropped outright.
- **The template's weightage ≠ the data's weightage.** The assumed-vs-empirical panel shows which
  factors the proforma over-weights (assumed rank ≫ empirical rank) and under-weights.
- **The composite score's predictive power** is whatever §5's r-values say — if `cumulative_site_score`
  has a near-zero r, the ranking it produces isn't tracking real volume.

These reproduce/refine the known internal finding that capacity-type factors (e.g. pay stations)
carry more real signal than the equally-weighted composite implies, and that proformas tend to
mis-state absolute volume.""")

code("""# one compact scorecard
print('='*64)
print('BACKTEST SCORECARD'.center(64))
print('='*64)
d = site.dropna(subset=['vol_monthly_y5','act_wash'])
ratio = (d['act_wash']/d['vol_monthly_y5']).replace([np.inf,-np.inf],np.nan).dropna()
print(f'matched sites analysed         : {len(site)}')
print(f'proj vs actual (mature)  r     : {stats.pearsonr(d.vol_monthly_y5,d.act_wash)[0]:.2f}')
print(f'median actual/projected ratio  : {ratio.median():.2f}')
print(f'median abs % error vs actual   : {(np.abs(d.act_wash-d.vol_monthly_y5)/d.act_wash).median()*100:.0f}%')
print(f'sites over-projected           : {(ratio<1).mean()*100:.0f}%')
top = corr_wash.head(3)
print('top-3 drivers of real wash     :', ', '.join(f'{i}({r:.2f})' for i,r in top['pearson_r'].items()))
print(f'cumulative_site_score  r       : {site[[\"cumulative_site_score\",\"act_wash\"]].dropna().corr().iloc[0,1]:.2f}')
print(f'factor->wash linear R^2        : {r2_lin:.2f}')
print('='*64)""")

nb['cells'] = cells
out = os.path.join(os.path.dirname(__file__), '..', 'proforma_backtest.ipynb')
nbf.write(nb, out)
print('wrote', out, 'with', len(cells), 'cells')
