"""Assemble the proforma backtest analysis notebook.

Regenerate + execute with:
    conda run -n sonnys python code/make_notebook.py
    conda run -n sonnys jupyter nbconvert --to notebook --execute --inplace proforma_backtest.ipynb

Kept as a script so the notebook is reproducible.  Every output cell is followed
by an **Insights** markdown cell derived from that output (see the repo skill
`.claude/skills/output-insights`).
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# Proforma Backtest — do the site-selection factors & projections match reality?

**Question.** Sonny's proformas score a prospective site on ~10 site-selection factors plus
demographics, roll them into target capture scores, and project a 5-year wash volume. The
projection is mechanically **`vol_yearly_yN = traffic_count x target_score_yN x ~300 operating
days`** (verified below, r=0.997) — so the whole backtest decomposes into: was the *capture*
assumption right, and do the *factors* that set it actually move real volume?

We address:

1. **Assumption vs reality** — projected vs actual volume, aligned by operating year from the
   real open date.
2. **Signal** — which factor scores/choices genuinely correlate with real performance
   (permutation p-values + Benjamini-Hochberg FDR, so 16 tested drivers can't produce cheap stars).
3. **Combinations** — are there factor *combos* with outsized wash-counts, or is that fluke?
   (max-statistic permutation over the searched combos).
4. **Weightage & predictive power** — the template's assumed weights vs empirical signal, and an
   honest *out-of-sample* test of what the inputs can predict.

Data: `old-proforma-combined-monthly.csv` / `old-proforma-combined.csv` (121 proformas, 115
address-matched to the actuals panel; operator-handoff sites are stitched across client_ids and
`match_operational_start` is the earliest segment's open; 3 near-dead panel records are flagged
`actuals_suspect` and excluded).""")

code("""import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
pd.set_option('display.width', 220, 'display.max_columns', 60)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .3, 'font.size': 10})
rng = np.random.default_rng(0)

import os as _os
BASE = '.' if _os.path.exists('old-proforma-combined-monthly.csv') else '..'
m = pd.read_csv(f'{BASE}/old-proforma-combined-monthly.csv')
s = pd.read_csv(f'{BASE}/old-proforma-combined.csv')
print('monthly rows:', len(m), '| proformas:', s.source_file.nunique(),
      '| matched:', s.match_status.str.startswith('matched').sum(),
      '| suspect actuals excluded:', (s.actuals_suspect == True).sum())

# the projection identity: vol = traffic x target x operating days
d0 = s.dropna(subset=['traffic_count', 'first_year_target_score', 'vol_yearly_y1'])
d0 = d0[(d0.traffic_count > 0) & (d0.first_year_target_score > 0)]
days = d0.vol_yearly_y1 / (d0.traffic_count * d0.first_year_target_score)
r_id = np.corrcoef(np.log(d0.traffic_count * d0.first_year_target_score), np.log(d0.vol_yearly_y1))[0, 1]
print(f'projection identity check: log(traffic x y1 target) vs log(vol_yearly_y1) r={r_id:.3f}; '
      f'implied operating days median={days.median():.0f} (IQR {days.quantile(.25):.0f}-{days.quantile(.75):.0f})')""")

md("""**Insights**
- The identity holds at r=0.997 with ~300 implied operating days: the projected volume is *literally*
  `traffic x capture-target x days`. Nothing else enters — so any projection error is a traffic-count
  error, a capture-target error, or the linear-in-traffic assumption itself (tested in §7).
- Because target scores are built from the factor+demographic scores, "do the factors work?" and
  "is the projection right?" are the same question asked at two grains.""")

md("""## 1. Site-level analysis table

Grain: one row per proforma (= one prospective site). From the clean (non-imputed) monthly
actuals of the matched site we derive:

- **`mature_wash`** — median wash-count of the last 12 clean months, **only accepted as "mature"
  if the site is ≥24 months old** at its last observation (`mature_ok`). Without the age gate a
  2023-24 opener's "last 12 months" is still ramp, which *overstates* over-projection when compared
  against the proforma's mature (Y5) target.
- **`y2_wash`** — median wash-count over operating months 12-23 (a uniform-age outcome for
  cross-site factor comparisons; needs an observed open date and ≥6 months in the window).
- **actual operating-year means** `mean_y1..y5` (+ observed-month counts `size_y1..y5`) aligned to
  `match_operational_start`, for sites whose open date falls **inside** the panel window (month-precise:
  `open_idx > 2020-01` and not before the site's first panel row — avoids the 2020 left-censor).
- Choice labels are normalized (case/typo variants like `MUTIPLE IN 4 MILES`, `2.0` collapse).""")

code("""def ym(x):
    try:
        mo, yr = str(x).split('-'); return int(yr) * 12 + int(mo)
    except Exception:
        return np.nan

site = s[s.match_status.str.startswith('matched') & (s.actuals_suspect != True)].copy()
site['open_idx'] = site.match_operational_start.map(ym)

mm = m[m.year.notna() & (m.imputed == 0)].copy()          # clean months only
mm = mm[mm.source_file.isin(site.source_file)]
mm['cal_idx'] = mm.year.astype(int) * 12 + mm.month.astype(int)
mm = mm.merge(site[['source_file', 'open_idx']], on='source_file')
mm['mo_open'] = mm.cal_idx - mm.open_idx                   # 0 = opening month
mm['op_year'] = mm.mo_open // 12 + 1
mm['blended_asp'] = mm.revenue / mm.wash_count.replace(0, np.nan)

last12 = (mm.sort_values(['source_file', 'cal_idx']).groupby('source_file')
          .apply(lambda g: pd.Series({'mature_wash': g.tail(12).wash_count.median(),
                                      'mature_rev': g.tail(12).revenue.median(),
                                      'mature_asp': g.tail(12).blended_asp.median(),
                                      'mature_n': len(g.tail(12)),
                                      'mature_mid_year': g.tail(12).year.mean()}),
                 include_groups=False))
site = site.merge(last12, left_on='source_file', right_index=True, how='left')

y2w = (mm[mm.mo_open.between(12, 23)].groupby('source_file').wash_count
       .agg(y2_wash='median', y2_n='size'))
site = site.merge(y2w, left_on='source_file', right_index=True, how='left')

opy = (mm[mm.op_year.between(1, 5)].groupby(['source_file', 'op_year']).wash_count
       .agg(['mean', 'size']).unstack())
opy.columns = [f'{a}_y{int(b)}' for a, b in opy.columns]
site = site.merge(opy, left_on='source_file', right_index=True, how='left')

PANEL_START = 2020 * 12 + 1
fc = mm.groupby('source_file').cal_idx.min(); lc = mm.groupby('source_file').cal_idx.max()
site = site.merge(fc.rename('first_cal'), left_on='source_file', right_index=True, how='left')
site = site.merge(lc.rename('last_cal'), left_on='source_file', right_index=True, how='left')
site['open_observed'] = (site.open_idx > PANEL_START) & (site.open_idx >= site.first_cal - 1)
site['age_mo'] = site.last_cal - site.open_idx + 1
site['mature_ok'] = (site.age_mo >= 24) & (site.mature_n >= 6) & (site.mature_wash > 0)
site['y2_ok'] = site.open_observed & (site.y2_n >= 6) & (site.y2_wash > 0)

def norm_choice(v):
    if pd.isna(v): return np.nan
    return str(v).upper().strip().replace('MUTIPLE', 'MULTIPLE').replace('2.0', '2')
for c in [c for c in site.columns if c.startswith('factor_') and c.endswith('_choice')]:
    site[c + '_n'] = site[c].map(norm_choice)

FS = [c for c in site.columns if c.startswith('factor_') and c.endswith('_score')]
FS = [c for c in FS if site[c].std() > 1e-9]     # weekly_hours is constant -> dropped
DEMO = ['cumulative_site_score', 'cumulative_demographic_score',
        'demog_avg_household_size_value', 'demog_pct_pop_25_65_value',
        'demog_pct_hh_income_35k_value', 'demog_base_price_carwash_value', 'traffic_count']

print(f'analysable proformas (matched, non-suspect): {len(site)}')
print(f'open date observed inside panel: {site.open_observed.sum()}')
print(f'mature outcome usable (>=24mo old, >=6 clean months): {site.mature_ok.sum()}')
print(f'uniform year-2 outcome usable: {site.y2_ok.sum()}')
print(f'live factor scores: {len(FS)} (weekly_hours constant across all proformas -> no signal possible)')
print('proforma types:', site.proforma_type.value_counts().to_dict())""")

md("""**Insights**
- 112 analysable sites, but each question gets its own honest denominator: 96 have an observed
  (non-censored) open date, **70** qualify for the mature comparison, **78** for the uniform
  year-2 outcome. Numbers below always state which n they use.
- `weekly_hours` cannot be backtested at all — every proforma ticked "More Than 70 Hours". A factor
  the template never varies carries zero information about site choice.
- ~92% of matched proformas are Express Exterior — findings are effectively about the express
  format; the few Flex/full-serve sites are checked as a robustness case in §3.""")

md("""## 2. Assumption vs reality — projected vs actual volume, aligned by operating year

Each proforma projects a *specific* volume per operating year (`vol_monthly_y1..y5`), so the fair
comparison is **year-aligned**: actual operating-year-N monthly washes (from the real open date,
≥10 observed months in that year) vs `vol_monthly_yN` — one panel per year. The sixth panel is the
**mature** check: age-gated actual (≥24 months old) vs `vol_monthly_y5`, because Y5 *is* the
template's mature target — that panel intentionally compares sites of mixed ages against the
mature level, the five year panels don't. Hover any dot for the site.""")

code("""import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'plotly_mimetype+notebook'

panels = []
for y in range(1, 6):
    dd = site[site.open_observed & site[f'size_y{y}'].ge(10) & site[f'mean_y{y}'].gt(0)
              ].dropna(subset=[f'vol_monthly_y{y}'])
    panels.append((f'Y{y}', dd[f'vol_monthly_y{y}'], dd[f'mean_y{y}'], dd))
dmat = site[site.mature_ok].dropna(subset=['vol_monthly_y5'])
panels.append(('Mature (>=24mo) vs Y5 target', dmat.vol_monthly_y5, dmat.mature_wash, dmat))

titles = []
for name, px, py, dd in panels:
    titles.append(f'{name} — n={len(dd)}, act/proj med={(py / px).median():.2f}, '
                  f'rho={stats.spearmanr(px, py)[0]:.2f}')

fig = make_subplots(rows=2, cols=3, subplot_titles=titles,
                    horizontal_spacing=.07, vertical_spacing=.16)
for k, (name, px, py, dd) in enumerate(panels):
    r, c = k // 3 + 1, k % 3 + 1
    cd = [[str(a), str(b), f'{v:.2f}'] for a, b, v in
          zip(dd.match_client_name.fillna(''), dd.address.fillna(''), py / px)]
    fig.add_trace(go.Scatter(
        x=px, y=py, mode='markers', customdata=cd,
        hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>'
                      'projected %{x:.0f}/mo - actual %{y:.0f}/mo (ratio %{customdata[2]})<extra></extra>',
        marker=dict(size=7, color='#4C72B0', opacity=.75, line=dict(width=.5, color='black')),
        showlegend=False), r, c)
    lim = float(max(px.max(), py.max())) * 1.05
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode='lines',
                             line=dict(color='red', dash='dash', width=1),
                             hoverinfo='skip', showlegend=False), r, c)
    fig.update_xaxes(range=[0, lim], row=r, col=c,
                     title_text='projected washes/mo' if r == 2 else None)
    fig.update_yaxes(range=[0, lim], row=r, col=c,
                     title_text='actual washes/mo' if c == 1 else None)
fig.update_annotations(font_size=11)
fig.update_layout(height=660, width=1140, template='plotly_white', font=dict(size=11),
                  title='Projected vs actual monthly washes, per operating year (red dash = perfect forecast)',
                  margin=dict(t=90))
fig.show()

ratio = dmat.mature_wash / dmat.vol_monthly_y5
print(f'mature (age-gated, n={len(dmat)}): actual/projected median {ratio.median():.2f} '
      f'IQR [{ratio.quantile(.25):.2f}, {ratio.quantile(.75):.2f}] | '
      f'over-projected {(ratio < 1).mean()*100:.0f}% | '
      f'spearman {stats.spearmanr(dmat.vol_monthly_y5, dmat.mature_wash)[0]:.2f}')
d_all = site[(site.mature_n >= 6) & (site.mature_wash > 0)].dropna(subset=['vol_monthly_y5'])
print(f'[no age gate, n={len(d_all)}: median ratio {(d_all.mature_wash / d_all.vol_monthly_y5).median():.2f} '
      f'- ramping sites drag it down; the gated number above is the honest one]')""")

md("""**Insights**
- **Year-aligned, the ramp years are only mildly hot at the median** (Y1 0.87 n=88, Y2 0.93 n=60)
  but the over-projection deepens with age: Y3 0.85, Y4 0.85, Y5 0.82, and the mature panel lands
  at **0.71 with 73% of 70 sites below the line** — the bias sits in the mature capture target,
  not the ramp shape.
- Rank signal is weak everywhere: within-year spearman runs **0.25 (Y1), 0.37 (Y2), 0.30 (Y3)**,
  then dies where the sample thins (Y4 0.38 on n=21, p≈0.09; Y5 0.22 on n=12, ns). Even at its
  best, the projection explains little of *which* site out-washes which.
- The vertical scatter around the red line dwarfs the bias in every panel — sites projected
  ~9-10k/mo realize anywhere from ~2k to ~15k (hover the outliers: they are real sites, not data
  errors). A flat haircut fixes the median, not the underwriting risk.
- Only the sixth panel mixes ages by design (mature actual vs the Y5 mature target); the five
  year panels are the like-for-like view this section previously lacked.""")

code("""rows = []
for y in range(1, 6):
    dd = site[site.open_observed & site.get(f'size_y{y}', pd.Series(dtype=float)).ge(10)
              & site.get(f'mean_y{y}', pd.Series(dtype=float)).gt(0)].dropna(subset=[f'vol_monthly_y{y}'])
    if len(dd) < 5: continue
    rr = dd[f'mean_y{y}'] / dd[f'vol_monthly_y{y}']
    rows.append({'op_year': f'Y{y}', 'n(>=10mo obs)': len(dd),
                 'proj_median': dd[f'vol_monthly_y{y}'].median(),
                 'actual_median': dd[f'mean_y{y}'].median(),
                 'ratio_q25': rr.quantile(.25), 'ratio_median': rr.median(), 'ratio_q75': rr.quantile(.75),
                 'over_projected_%': (rr < 1).mean() * 100,
                 'MdAPE_%': ((dd[f'mean_y{y}'] - dd[f'vol_monthly_y{y}']).abs() / dd[f'mean_y{y}']).median() * 100,
                 'spearman': stats.spearmanr(dd[f'vol_monthly_y{y}'], dd[f'mean_y{y}'])[0]})
acc = pd.DataFrame(rows)
print('Projection accuracy by operating year (observed opens, months aligned to real open date):')
print(acc.round(2).to_string(index=False))

# where does the miss come from? actual capture vs the assumed target
d2 = site[site.mature_ok & site.traffic_count.gt(0)].copy()
act_capture = d2.mature_wash * 12 / (d2.traffic_count * 300)
print(f'\\ncapture rate (share of daily passing traffic washed, mature): '
      f'actual median {act_capture.median()*100:.2f}% vs assumed mature target {d2.mature_target_score.median()*100:.2f}% '
      f'-> ratio {(act_capture / d2.mature_target_score).median():.2f} (n={len(d2)})')

d = site[site.mature_ok].dropna(subset=['vol_monthly_y5']).copy()
d['cohort'] = np.where(d.match_operational_start.str[-4:].astype(int) <= 2021, 'opened 2020-21', 'opened 2022+')
print('\\nmature actual/projected by open cohort:')
print(d.groupby('cohort').apply(lambda g: pd.Series(
    {'n': len(g), 'ratio_median': (g.mature_wash / g.vol_monthly_y5).median()}),
    include_groups=False).round(2).to_string())""")

md("""**Insights**
- **The ramp shape is roughly right; the mature level is set too high.** Year-aligned medians run
  0.87 (Y1), 0.93 (Y2) then drift down 0.85 → 0.82 by Y5 with 60→75% of sites over-projected:
  the template's error grows exactly where its capture target steps up to "mature".
- The whole mature miss is in the **capture assumption**: actual mature capture is ~0.94% of daily
  traffic vs the assumed ~1.43% — ratio 0.70, i.e. the same 30% the volume ratio shows. Traffic
  counts aren't the (average) problem; the target-score calibration is.
- MdAPE is 30-49% per operating year — even in Y1-Y2, where the *median bias* is small, individual
  sites routinely miss by a third or more; the bias correction fixes the mean, not the spread.
- 2020-21 openers backtest worse (0.61 vs 0.76) — consistent with COVID-era openings and with those
  proformas being older template vintages; treat era as a confounder, not a conclusion.""")

md("""## 3. Which inputs carry real signal?

Spearman rank correlation of every live input against two outcomes — **log mature washes**
(age-gated, n=70) and **log year-2 washes** (uniform age, n=78) — with **permutation p-values**
(20k shuffles) and **BH-FDR q** across the 16 tested drivers. `partial_r` re-tests each driver
after residualizing both sides on **log traffic** ("does it add signal *beyond* traffic?").""")

code("""def perm_spearman(x, y, n=20000):
    ok = x.notna() & y.notna()
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    if len(x) < 10 or np.std(x) == 0: return np.nan, np.nan, len(x)
    xr, yr = stats.rankdata(x), stats.rankdata(y)
    xr = (xr - xr.mean()) / xr.std(); yr = (yr - yr.mean()) / yr.std()
    r0 = float(np.mean(xr * yr))
    perm = np.array([np.mean(xr * rng.permutation(yr)) for _ in range(n)])
    return r0, max((np.abs(perm) >= abs(r0)).mean(), 1 / n), len(x)

def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = ~np.isnan(p)
    pv = p[ok]; n = len(pv); o = np.argsort(pv)
    r = np.minimum.accumulate((pv[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(r, 1); q[ok] = out
    return q

results = {}
for name, mask, col in [('mature', site.mature_ok, 'mature_wash'), ('year2', site.y2_ok, 'y2_wash')]:
    sub = site[mask]; ylog = np.log(sub[col])
    lt = np.log(sub.traffic_count.replace(0, np.nan))
    res = []
    for f in FS + DEMO:
        r, p, n = perm_spearman(sub[f], ylog)
        pr = np.nan
        okp = sub[f].notna() & ylog.notna() & lt.notna()
        if okp.sum() >= 10 and sub.loc[okp, f].std() > 0 and f != 'traffic_count':
            ry = ylog[okp] - np.polyval(np.polyfit(lt[okp], ylog[okp], 1), lt[okp])
            rf = sub.loc[okp, f] - np.polyval(np.polyfit(lt[okp], sub.loc[okp, f], 1), lt[okp])
            pr = stats.spearmanr(rf, ry)[0]
        res.append({'driver': f, 'spearman_r': r, 'perm_p': p, 'partial_r_ctrl_traffic': pr, 'n': n})
    R = pd.DataFrame(res); R['fdr_q'] = bh(R.perm_p)
    results[name] = R.sort_values('spearman_r', key=lambda v: v.abs(), ascending=False)

print('--- outcome: log MATURE washes (age-gated) ---')
print(results['mature'].round(3).to_string(index=False))
print()
print('--- outcome: log YEAR-2 washes (uniform age) ---')
print(results['year2'].round(3).to_string(index=False))""")

code("""R = results['mature'].set_index('driver').sort_values('spearman_r')
fig, ax = plt.subplots(figsize=(9, 6.5))
colors = ['#C44E52' if v < 0 else '#55A868' for v in R.spearman_r]
ax.barh(R.index, R.spearman_r, color=colors, edgecolor='k', lw=.4)
ax.scatter(R.partial_r_ctrl_traffic, range(len(R)), color='k', s=20, zorder=3,
           label='partial r (traffic controlled)')
for i, (idx, row) in enumerate(R.iterrows()):
    if row.fdr_q < 0.05:
        ax.text(row.spearman_r + np.sign(row.spearman_r) * .012, i, '*',
                va='center', ha='left' if row.spearman_r >= 0 else 'right', fontsize=15)
ax.axvline(0, color='k', lw=.8)
ax.set(xlabel='Spearman r vs log mature washes   (* = survives BH-FDR q<0.05, n=70)',
       title='Which proforma inputs actually track real wash volume?')
ax.legend(loc='lower right'); plt.tight_layout(); plt.show()

ex = site[site.mature_ok & site.proforma_type.str.contains('Express', na=False)]
print('robustness (Express-only, n=%d):' % len(ex))
for f in ['factor_pay_stations_score', 'factor_free_vacuum_slots_score', 'factor_type_of_site_score']:
    r, p = stats.spearmanr(ex[f], np.log(ex.mature_wash))
    print(f'  {f:34s} r={r:.2f} p={p:.4f}')
sub = site[site.mature_ok]
r_era, p_era = stats.spearmanr(sub.mature_mid_year, np.log(sub.mature_wash))
print(f'era check: mature-window calendar year vs outcome r={r_era:.2f} p={p_era:.2f} '
      '(factor signal is not a calendar-era artifact)')""")

md("""**Insights**
- **Four inputs survive FDR on the mature outcome — pay stations (r=0.41, q=0.005), free-vacuum
  slots (0.34, q=0.025), type-of-site (0.33, q=0.025) and the composite site score (0.32,
  q=0.028)** — and the first two replicate on the independent year-2 outcome (q=0.017/0.028).
  This confirms the prior internal finding (proforma_db study: "only pay stations predicts") and
  extends it: vacuums and corner/light siting are real too.
- **Traffic count itself does NOT survive** (r≈0.18-0.21, q≈0.28-0.35) — remarkable given the
  projection multiplies everything by it. The capacity factors keep partial r ≈ 0.30 after
  controlling for traffic: they are not proxies for busier roads.
- Demographics are dead weight in this sample (|r| ≤ 0.15, q ≥ 0.53 for all four), and
  `nearest_competition` even points the *wrong* way (more competition ↔ slightly more washes —
  co-location with retail gravity, not a moat).
- Signal, not fluke: the starred factors are FDR-controlled, replicate across two outcome
  definitions, hold Express-only (r 0.37-0.41), and the era check (r=0.10, p=0.41) rules out a
  calendar confound. But note the *sign* of causality is untested — operators may build more pay
  stations where they expect more volume.""")

md("""## 4. Choice-level reality — what does each ticked box mean in washes/month?

Median mature washes per (normalized) choice level, cells with n≥5, with a permutation
Kruskal-Wallis p per factor (does the choice split performance at all?).""")

code("""sub = site[site.mature_ok].copy()
def kw_perm(groups, n=5000):
    obs = stats.kruskal(*groups)[0]
    pool = np.concatenate(groups); sizes = [len(g) for g in groups]; cnt = 0
    for _ in range(n):
        pm = rng.permutation(pool); i = 0; gs = []
        for szz in sizes: gs.append(pm[i:i + szz]); i += szz
        if stats.kruskal(*gs)[0] >= obs: cnt += 1
    return max(cnt / n, 1 / n)

for c in ['factor_pay_stations_choice_n', 'factor_free_vacuum_slots_choice_n',
          'factor_type_of_site_choice_n', 'factor_entrance_stack_up_choice_n',
          'factor_nearest_competition_choice_n', 'factor_visibility_choice_n',
          'factor_area_profile_choice_n', 'factor_traffic_speed_choice_n']:
    g = sub.groupby(c).mature_wash.agg(['median', 'count'])
    g = g[g['count'] >= 5].sort_values('median', ascending=False)
    if len(g) < 2: continue
    p = kw_perm([sub.loc[sub[c] == lev, 'mature_wash'].to_numpy() for lev in g.index])
    print(f"{c.replace('factor_', '').replace('_choice_n', '').upper():24s} KW perm p={p:.3f}")
    print(g.round(0).to_string(), end='\\n\\n')""")

md("""**Insights**
- **Capacity ladders are monotone and huge**: pay stations 1 → 2 → 3+ gives 3.3k → 6.1k → 9.4k
  washes/mo (KW p≈0.02); vacuums <12 → 12-20 → >20 gives 3.6k → 6.3k → 9.1k (p≈0.01). Each step
  up the ladder roughly +50%. (3+ pay stations is n=6 — direction is solid, the 9.4k magnitude is
  an anecdote.)
- **The traffic light is worth more than the corner**: corner+light 9.8k > inside-near-light 5.9k >
  corner-no-light 4.8k > inside-no-light 3.3k (p≈0.03). "With light" beats "without" *within both*
  lot types; the template scores corner-without-light above inside-near-light — the data says
  that's backwards.
- Entrance stack-up's low p (≈0.007) is **not trustworthy as a ladder**: it's non-monotone (deepest
  stack "More than 20 vehicles" has the *lowest* median, 3.5k) and its top cell is the ambiguous
  double-ticked "20-15 / Less than 10" extraction artifact (n=6). Read it as "something correlates
  here", not "build deeper stacks".
- Competition, visibility, area profile, traffic speed: p = 0.5-0.9 — the boxes get ticked, the
  washes don't move. These four (plus constant weekly-hours) are ~half the scorecard doing ~nothing.""")

md("""## 5. Combinations — do certain factor combos have outsized wash-counts, or is that fluke?

All 2-factor combinations of the binarized levers (pay≥2, vacuums high, corner, light, stack high,
multi-competitor-2mi) are searched for the biggest median lift vs the rest. Because we *searched*,
significance is judged by **max-statistic permutation**: shuffle outcomes 3000x, rerun the *entire
search* each time, and ask how often the best chance-combo beats the best real one.""")

code("""sub = site[site.mature_ok].copy()
sub['pay2p'] = sub.factor_pay_stations_choice_n.isin(['2', '3 OR MORE'])
sub['vac_hi'] = sub.factor_free_vacuum_slots_choice_n.isin(['12 - 20 VEHICLES', 'MORE THAN 20 VEHICLES'])
sub['corner'] = sub.factor_type_of_site_choice_n.str.contains('CORNER', na=False)
sub['light'] = sub.factor_type_of_site_choice_n.str.contains('WITH LIGHT|NEAR LIGHT', na=False, regex=True)
sub['stack_hi'] = sub.factor_entrance_stack_up_choice_n.isin(['20 - 15 VEHICLES', 'MORE THAN 20 VEHICLES'])
sub['comp_multi2'] = sub.factor_nearest_competition_choice_n.eq('MULTIPLE IN 2 MILES')
FLAGS = ['pay2p', 'vac_hi', 'corner', 'light', 'stack_hi', 'comp_multi2']
yv = sub.mature_wash.to_numpy()

pairs = []
for i in range(len(FLAGS)):
    for j in range(i + 1, len(FLAGS)):
        mk = (sub[FLAGS[i]] & sub[FLAGS[j]]).to_numpy()
        if mk.sum() >= 8 and (~mk).sum() >= 8:
            pairs.append((f'{FLAGS[i]} & {FLAGS[j]}', mk,
                          np.median(yv[mk]) - np.median(yv[~mk])))
print('single levers first (median washes/mo in vs out):')
for f in FLAGS:
    mk = sub[f].to_numpy()
    if mk.sum() >= 8 and (~mk).sum() >= 8:
        print(f'  {f:12s} n={mk.sum():3d}  in={np.median(yv[mk]):6.0f}  out={np.median(yv[~mk]):6.0f}  '
              f'lift={np.median(yv[mk]) - np.median(yv[~mk]):+6.0f}')
print('\\ntop 2-factor combos by |median lift|:')
for name, mk, lift in sorted(pairs, key=lambda t: -abs(t[2]))[:5]:
    print(f'  {name:22s} n={mk.sum():3d}  in={np.median(yv[mk]):6.0f}  out={np.median(yv[~mk]):6.0f}  lift={lift:+6.0f}')

obs_best = max(abs(t[2]) for t in pairs); masks = [t[1] for t in pairs]
hits = 0; NP = 3000
for _ in range(NP):
    yp = rng.permutation(yv)
    if max(abs(np.median(yp[mk]) - np.median(yp[~mk])) for mk in masks) >= obs_best: hits += 1
print(f'\\nbest combo lift = {obs_best:.0f} washes/mo; max-statistic permutation p = {hits/NP:.3f} '
      f'(chance of a lift this big somewhere in the same {len(pairs)}-combo search)')""")

md("""**Insights**
- The best combo (**pay≥2 & vacuums-high: 7.0k vs 3.3k washes/mo, +3.7k**) looks dramatic, but the
  honest search-corrected verdict is **p≈0.10 — not proof of a special combination**. And the single
  lever `vac_hi` alone already gives +3.6k: the combos add essentially nothing beyond their strongest
  member. **Additive capacity main-effects, no detectable synergy at n=70.**
- Practical read: "2+ pay stations AND high vacuum count" is a fine *screen* because each half is
  independently real (§3), not because the pair is magic.
- A competitor-density combo never rises above +0.7k — consistent with competition carrying no
  signal on this sample.""")

md("""## 6. Weightage & honest predictive power

Left: the template's realized weight per factor (its mean |score| share of the cumulative site
score) vs the data's univariate signal (|Spearman| vs mature washes, normalized to shares).
Right: **out-of-sample** R² (5-fold x 40 repeats, ridge) — what the inputs can actually *predict*.""")

code("""assumed = site[FS].abs().mean(); assumed = assumed / assumed.sum()
emp = results['mature'].set_index('driver').loc[FS, 'spearman_r'].abs()
emp = emp / emp.sum()
w = pd.DataFrame({'assumed_weight_share': assumed, 'empirical_signal_share': emp}).sort_values('assumed_weight_share')

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
idx = np.arange(len(w))
ax[0].barh(idx - .2, w.assumed_weight_share, height=.4, label='assumed (template weight)', color='#8172B3')
ax[0].barh(idx + .2, w.empirical_signal_share, height=.4, label='empirical (|spearman| share)', color='#CCB974')
ax[0].set_yticks(idx); ax[0].set_yticklabels([i.replace('factor_', '').replace('_score', '') for i in w.index])
ax[0].set(xlabel='share', title='Template weight vs data signal, per factor'); ax[0].legend(fontsize=8)

d = site[site.mature_ok].dropna(subset=FS + ['traffic_count', 'cumulative_site_score',
                                             'cumulative_demographic_score', 'vol_monthly_y5'])
d = d[d.traffic_count > 0]
ylog = np.log(d.mature_wash).to_numpy()
models = {
    'traffic only': np.log(d[['traffic_count']]),
    'site score only': d[['cumulative_site_score']],
    'score + log traffic': pd.concat([d[['cumulative_site_score']], np.log(d[['traffic_count']])], axis=1),
    'capacity trio': d[['factor_pay_stations_score', 'factor_free_vacuum_slots_score',
                        'factor_entrance_stack_up_score']],
    'all 9 factor scores': d[FS],
    'proforma projection (log vol_y5)': np.log(d[['vol_monthly_y5']]),
}
cv = RepeatedKFold(n_splits=5, n_repeats=40, random_state=0)
names, r2s = [], []
for name, X in models.items():
    Xv = X.to_numpy(); sc = []
    for tr, te in cv.split(Xv):
        mu, sd = Xv[tr].mean(0), Xv[tr].std(0) + 1e-12
        mdl = Ridge(alpha=1.0).fit((Xv[tr] - mu) / sd, ylog[tr])
        pred = mdl.predict((Xv[te] - mu) / sd)
        sc.append(1 - ((ylog[te] - pred) ** 2).sum() / ((ylog[te] - ylog[tr].mean()) ** 2).sum())
    names.append(name); r2s.append(np.mean(sc))
    print(f'  {name:34s} oos R^2 = {np.mean(sc):+.3f}')
ax[1].barh(names, r2s, color=['#55A868' if v > 0 else '#C44E52' for v in r2s], edgecolor='k', lw=.4)
ax[1].axvline(0, color='k', lw=.8)
ax[1].set(xlabel='out-of-sample R^2 (log mature washes)', title=f'What can the inputs predict? (n={len(d)})')
plt.tight_layout(); plt.show()""")

md("""**Insights**
- **The template spreads weight nearly evenly; the data concentrates it.** Pay stations, vacuums and
  type-of-site earn 2-3x their assumed share of the signal; competition, visibility, accessibility
  and traffic-speed earn well under theirs.
- **Nothing here is a usable point-forecaster**: the best model (site score + log traffic) reaches
  oos R² ≈ 0.10; traffic alone ≈ 0.0; the proforma's own projection ≈ 0.03; and stuffing in all 9
  factor scores *hurts* (negative oos R² — pure overfit at n=70). The inputs weakly *rank* sites,
  they cannot *size* them.
- This is the quantitative case for the current cold-start approach: site-selection inputs alone
  cap out near R²≈0.1, so volume forecasting has to lean on comparable-site behaviour
  (neighbours/cluster anchors), with these factors as secondary adjusters at most.
- Composite beats its parts out-of-sample (score 0.065 vs capacity trio -0.09): with 70 sites, even
  real univariate signal is too noisy to re-weight reliably — another reason to *re-calibrate* the
  existing score rather than fit a new factor model on this sample.""")

md("""## 7. Where does the projection error come from?

Spearman of every input against **log(actual/projected)** mature volume (negative = over-projection),
FDR-corrected. If the projection formula were well-calibrated, *nothing* would correlate.""")

code("""sub = site[site.mature_ok].dropna(subset=['vol_monthly_y5']).copy()
sub['log_err'] = np.log(sub.mature_wash / sub.vol_monthly_y5)
res = []
for f in FS + DEMO:
    r, p, n = perm_spearman(sub[f], sub.log_err)
    res.append({'driver': f, 'spearman_r': r, 'perm_p': p, 'n': n})
E = pd.DataFrame(res); E['fdr_q'] = bh(E.perm_p)
print(E.sort_values('spearman_r', key=lambda v: v.abs(), ascending=False).round(3).head(8).to_string(index=False))
print(f'median log_err = {sub.log_err.median():.2f} (= ratio {np.exp(sub.log_err.median()):.2f})')

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.scatter(sub.traffic_count, sub.log_err, s=30, alpha=.65, edgecolor='k', lw=.3)
b = np.polyfit(np.log(sub.traffic_count), sub.log_err, 1)
xs = np.linspace(sub.traffic_count.min(), sub.traffic_count.max(), 60)
ax.plot(xs, np.polyval(b, np.log(xs)), 'r-', lw=1.4)
ax.axhline(0, color='k', lw=.8, ls='--')
ax.set(xscale='log', xlabel='traffic_count (daily, log scale)',
       ylabel='log(actual / projected)  [mature]',
       title='Over-projection grows with traffic: the linear-in-traffic assumption fails')
plt.tight_layout(); plt.show()""")

md("""**Insights**
- **Traffic is the strongest error driver (r=-0.35, FDR q≈0.04): the busier the road, the more the
  proforma over-projects.** Volume does not scale linearly with traffic — capture saturates (tunnel
  throughput, membership catchment) — yet the formula multiplies straight through. A concave
  traffic term (or a capture cap) would fix the *shape* of the bias, not just its level.
- `cumulative_site_score` correlates *positively* with the error ratio (r≈0.30, q≈0.09): highly-scored
  sites under-promise, low-scored sites over-promise — i.e. the score is directionally right but the
  target-score mapping under-uses it while traffic over-drives the output.
- Everything else washes out after FDR: the miss is not about demographics or any single checkbox —
  it's the two moving parts the formula leans on hardest (traffic level, capture target).""")

md("""## 8. ASP & price reality check

The proforma sets menu prices; the panel reports realized blended ASP (revenue / washes over the
mature window).""")

code("""d = site[site.mature_ok].dropna(subset=['pkg_menu1_price', 'mature_asp'])
d = d[(d.mature_asp > 0) & (d.mature_asp < 100)]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].scatter(d.pkg_menu1_price, d.mature_asp, s=30, alpha=.6, edgecolor='k', lw=.3)
lim = [0, max(d.pkg_menu1_price.max(), d.mature_asp.max()) * 1.05]
ax[0].plot(lim, lim, 'r--', lw=1)
ax[0].set(xlabel='proforma menu-1 price ($)', ylabel='actual blended ASP ($)',
          title='Priced vs realized ASP', xlim=lim, ylim=lim)
ax[1].hist(d.mature_asp - d.pkg_menu1_price, bins=20, color='#4C72B0', edgecolor='w')
ax[1].axvline(0, color='r', ls='--')
ax[1].set(xlabel='actual ASP - proforma menu-1 price ($)', ylabel='sites', title='ASP gap')
plt.tight_layout(); plt.show()
r, p = stats.spearmanr(d.pkg_menu1_price, d.mature_asp)
print(f'median realized ASP: {d.mature_asp.median():.2f} | median menu-1 price: {d.pkg_menu1_price.median():.2f} '
      f'| spearman(price, ASP)={r:.2f} (p={p:.3f}, n={len(d)})')""")

md("""**Insights**
- Realized blended ASP (median **$16.04**) runs ~60% above the menu-1 price (median **$10**) —
  membership mix and upgrades dominate realized revenue per car, so the priced menu is a floor,
  not a forecast. Volume, not price, is where proformas err optimistic.
- The priced menu carries **no cross-site signal for realized ASP** (spearman -0.15, p=0.22, n=65):
  what a site's menu says and what its cars actually pay are unrelated at this grain, because
  blended ASP is a mix variable, not a price variable.
- Caveat: blended ASP inherits the panel's known revenue-corruption modes on a minority of
  operators; the (0,100) trim and mature-window median blunt but don't eliminate that.""")

md("""## 9. Findings

1. **Bias, quantified honestly.** Mature volume over-projected ~30% at the median (0.71, 73% of 70
   sites), ramp years nearly unbiased (Y1 0.87, Y2 0.93) — the error is in the mature capture
   target (assumed 1.43% of traffic vs realized 0.94%), and it concentrates on high-traffic sites
   (linear-in-traffic fails; err vs traffic r=-0.35, q=0.04). MdAPE 30-49% per year regardless.
2. **The factors are not random guesses — but only some of them.** Pay stations, free-vacuum slots,
   type-of-site (traffic light > corner) and the composite score survive permutation + FDR and
   replicate on a second outcome; competition, visibility, accessibility, traffic-speed, area
   profile and all four demographics do not. Half the scorecard is dead weight on this sample.
3. **Combos are additive, not magic** — best pair (pay≥2 & vac-high, +3.7k/mo) fails the
   search-corrected permutation test (p≈0.10) and adds nothing beyond its strongest single lever.
4. **Ranking ≠ sizing.** Everything the template knows, combined, predicts log mature volume at
   oos R² ≈ 0.10; its own projection manages 0.03. Use the surviving factors as screens/adjusters;
   size volume from comparable-site actuals.""")

code("""print('=' * 66)
print('BACKTEST SCORECARD'.center(66))
print('=' * 66)
d = site[site.mature_ok].dropna(subset=['vol_monthly_y5'])
ratio = d.mature_wash / d.vol_monthly_y5
print(f'proformas matched / analysed mature : {len(site)} / {len(d)}')
print(f'median actual/projected (mature)    : {ratio.median():.2f}   (73% over-projected)')
print(f'ramp years Y1/Y2 median ratio       : see section 2 table (~0.87 / 0.93)')
print(f'assumed vs realized mature capture  : {d.mature_target_score.median()*100:.2f}% vs '
      f'{(d.mature_wash*12/(d.traffic_count*300)).median()*100:.2f}% of daily traffic')
Rm = results['mature']
sig = Rm[Rm.fdr_q < .05]
print('FDR-significant drivers of volume   : ' + ', '.join(
    f"{r.driver.replace('factor_','').replace('_score','')}({r.spearman_r:.2f})" for r in sig.itertuples()))
print(f'combo search (max-stat permutation) : best +3.7k washes/mo, p~0.10 -> no synergy proven')
print(f'best out-of-sample R^2 from inputs  : ~0.10 (score+traffic) | projection alone ~0.03')
print('=' * 66)""")

nb['cells'] = cells
out = os.path.join(os.path.dirname(__file__), '..', 'proforma_backtest.ipynb')
nbf.write(nb, out)
print('wrote', out, 'with', len(cells), 'cells')
