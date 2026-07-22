"""Assemble the proforma backtest notebook: OLD PROFORMA vs OUR MODEL vs ACTUAL.

Regenerate + execute with:
    conda run -n sonnys python code/make_notebook.py
    conda run -n sonnys jupyter nbconvert --to notebook --execute --inplace proforma_backtest.ipynb

Kept as a script so the notebook is reproducible. Every output cell is followed by an
**Insights** markdown cell derived from that output (repo skill: .claude/skills/output-insights).

Data (all committed):
  ensemble/results/ensemble_features.csv  one row per matched proforma site, aligning the three
      forecasters + actuals: pf_y1..5 (old proforma), cs_y1..5 / plateau_loo (our cold-start
      model, leave-one-out), act_y1..5 / mature_wash (ground truth), + capacity factor scores.
  old-proforma-combined.csv               the full site-factor score set (for §4 significance).
  ensemble/results/ensemble_r_blend.json  the leave-one-site-out ensemble scorecard row.
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# Proforma backtest — **old proforma vs our model vs actual**

Every built site had a pre-build **old-proforma projection (v1.0)**. We test it against two of
our own forecasts for the same pin (both leave-one-out — they never see the site's own history):
**Model v1.5**, the cold-start model from *location alone*, and **Model 5**, the super ensemble
that adds the site's own capacity inputs (**pay stations, vacuums, site type — taken from the
proforma's own data** — plus traffic). All are scored against the **actual** operating wash
counts. This notebook, simply:

1. **Accuracy** — how close did each forecast land to reality, year by year and at maturity? (§1–§3)
2. **What actually drives volume** — of the ~10 site-selection factors the proforma scores, do
   only **pay stations, vacuums and site type** carry real signal, and are the rest noise? (§4–§5)
3. **How good, really** — the spread behind the medians, a reality-check at scale (862 sites),
   and how far the forecast sharpens once a site opens. (§6–§8)

Colours are consistent throughout: **Proforma v1.0 = orange, Model v1.5 = blue, Model 5 = green,
actual = the reference (grey line / axis).**""")

code("""import json, os as _os
import numpy as np, pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'plotly_mimetype+notebook'
rng = np.random.default_rng(0)

BASE = '.' if _os.path.exists('ensemble/results/ensemble_features.csv') else '..'
E = pd.read_csv(f'{BASE}/ensemble/results/ensemble_features.csv')
S = pd.read_csv(f'{BASE}/old-proforma-combined.csv')
M5 = pd.read_csv(f'{BASE}/ensemble/results/model5_loso.csv')      # Model 5 leave-one-out forecasts

# consistent colours
C_PF, C_MODEL, C_M5, C_ACT = '#eb6834', '#2a78d6', '#1baf7a', '#6b6a63'   # proforma / v1.5 / Model 5 / actual
C_SIG, C_NS = '#1baf7a', '#c2c0b6'                            # factor significant / not

# merge the full factor-score set (for §4) + Model 5 forecasts onto the aligned comparison table
FAC = [c for c in S.columns if c.startswith('factor_') and c.endswith('_score')] + \\
      ['cumulative_site_score', 'demog_avg_household_size_value', 'demog_pct_pop_25_65_value',
       'demog_pct_hh_income_35k_value', 'demog_base_price_carwash_value']
CHO = ['factor_pay_stations_choice', 'factor_free_vacuum_slots_choice', 'factor_type_of_site_choice']
D = E.merge(S[['source_file'] + FAC + CHO], on='source_file', how='left').merge(M5, on='source_file', how='left')

def mdape(a, p):  a, p = np.asarray(a,float), np.asarray(p,float); return float(np.median(np.abs(a-p)/a)*100)
def within(a, p, b=.2): a, p = np.asarray(a,float), np.asarray(p,float); return float((np.abs(a/p-1)<=b).mean()*100)
def rho(a, p):    return float(stats.spearmanr(p, a)[0])
def ratio_med(a, p): a, p = np.asarray(a,float), np.asarray(p,float); return float(np.median(a/p))

MATURE = D[D.mature_ok & D.mature_wash.gt(0)].copy()          # age-gated (>=24mo), n=70
print(f'matched proforma sites: {len(D)} | mature (>=24mo) usable: {len(MATURE)} | '
      f'open-date observed: {int(D.open_observed.sum())}')""")

md("""**Insights**
- The comparison universe is **112 matched sites** (every old proforma we could address-match to
  a real operating site); the strict **mature** analysis uses the **70** that are ≥24 months old,
  so a still-ramping site is never scored against a mature target.
- Every number below is a like-for-like triple — same site, same operating year, all three of
  {old proforma, our model, actual} present — so the two forecasters are judged on identical
  ground, never on different site sets.
- Caveat: our model reads neighbours' actuals from the full panel (some of it after these sites
  opened), while the proforma was written ~18 months before opening — so our model's edge is an
  upper bound on its true head-start advantage.""")

md("""## 1. The three forecasters vs reality — at maturity

Each dot is one mature site: its **forecast** monthly washes (x) against what it **actually** did
(y). Left → right: old proforma, our location-only model (v1.5), and Model 5 (v1.5 + the site's
pay-station / vacuum / site-type inputs). Dots on the grey line = perfect; a tighter cloud sitting
*on* the line is a better forecaster. Hover any dot for the site.""")

code("""dm = MATURE.dropna(subset=['pf_y5', 'plateau_loo', 'm5_mature']).copy()
lim = float(max(dm.pf_y5.max(), dm.plateau_loo.max(), dm.m5_mature.max(), dm.mature_wash.max()) * 1.05)
panels = [('pf_y5', C_PF, 'Proforma v1.0'), ('plateau_loo', C_MODEL, 'Model v1.5 (location only)'),
          ('m5_mature', C_M5, 'Model 5 (＋ site factors)')]
fig = make_subplots(rows=1, cols=3, horizontal_spacing=.055, subplot_titles=[
    f'{nm}<br><span style="font-size:11px">MdAPE {mdape(dm.mature_wash, dm[c]):.0f}%'
    f' · bias {ratio_med(dm.mature_wash, dm[c]):.2f}</span>' for c, _, nm in panels])
for col, (xcol, colr, nm) in enumerate(panels, 1):
    cd = np.stack([dm.source_file.str.slice(0, 40), (dm.mature_wash/dm[xcol]).round(2)], axis=1)
    fig.add_trace(go.Scatter(x=dm[xcol], y=dm.mature_wash, mode='markers', customdata=cd,
        marker=dict(size=6, color=colr, opacity=.75, line=dict(width=.5, color='white')),
        hovertemplate='%{customdata[0]}<br>forecast %{x:.0f} · actual %{y:.0f} '
        '(ratio %{customdata[1]})<extra></extra>', showlegend=False), 1, col)
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode='lines', line=dict(color=C_ACT, dash='dash', width=1),
        hoverinfo='skip', showlegend=False), 1, col)
    fig.update_xaxes(range=[0, lim], title_text='forecast washes/mo', row=1, col=col)
    fig.update_yaxes(range=[0, lim], title_text='actual washes/mo' if col == 1 else None, row=1, col=col)
fig.update_layout(height=430, width=1050, template='plotly_white',
                  title=f'Forecast vs actual mature monthly washes (n={len(dm)})', margin=dict(t=95))
fig.show()
for c, _, nm in panels:
    print(f'{nm:32s} MdAPE {mdape(dm.mature_wash, dm[c]):4.0f}%   bias {ratio_med(dm.mature_wash, dm[c]):.2f}   '
          f'within +-20% {within(dm.mature_wash, dm[c]):3.0f}%')""")

md("""**Insights**
- **Adding the site factors is the whole ballgame.** Mature error falls **58% (proforma) → 46%
  (v1.5, location only) → 30% (Model 5)**, and the bias goes **0.71 → 0.73 → 1.00** — Model 5 is the
  first forecaster that is both accurate *and* unbiased; its cloud actually sits on the line.
- **This answers "how can a linear formula tie proper modelling?"** — it only ties the *location-only*
  model (v1.5), because neither knows the site's pay stations / vacuums / type. Give the model those
  three inputs (Model 5) and it wins outright: near-half the proforma's error.
- v1.5 over-projects on *these* older matched sites (bias 0.73) — a cohort quirk, not a model flaw
  (§7 shows it's unbiased on the full 862). Model 5's factor calibration corrects even that here.
- Caveat: even Model 5 misses a typical site by ~30% (and v1.0/v1.5 by ~46–58%) — maturity volume
  for a brand-new site is genuinely hard; none is a ±10% instrument (see §3, §6).""")

md("""## 2. Accuracy by operating year — how close is the wash count?

Aligned to each site's real open date (only years with ≥10 observed months). Left = **typical
error** (MdAPE, lower is better); right = **hit-rate** (share of sites within ±20% of forecast,
higher is better). Both are pure count-accuracy — is the predicted number right? Orange = proforma,
blue = v1.5, green = Model 5.""")

code("""rows = []
for y in range(1, 6):
    d = D[D.open_observed & D[f'nobs_y{y}'].ge(10) & D[f'act_y{y}'].gt(0)
          & D[f'pf_y{y}'].gt(0) & D[f'cs_y{y}'].gt(0) & D[f'm5_y{y}'].gt(0)]
    if len(d) < 8: continue
    a = d[f'act_y{y}']
    rows.append(dict(yr=f'Y{y}', n=len(d),
        pf_e=mdape(a, d[f'pf_y{y}']), md_e=mdape(a, d[f'cs_y{y}']), m5_e=mdape(a, d[f'm5_y{y}']),
        pf_w=within(a, d[f'pf_y{y}']), md_w=within(a, d[f'cs_y{y}']), m5_w=within(a, d[f'm5_y{y}']),
        pf_r=rho(a, d[f'pf_y{y}']), md_r=rho(a, d[f'cs_y{y}']), m5_r=rho(a, d[f'm5_y{y}'])))
A = pd.DataFrame(rows)
xa = [f"{r.yr}<br><span style='font-size:10px;color:#999'>n={r.n}</span>" for r in A.itertuples()]
fig = make_subplots(rows=1, cols=2, horizontal_spacing=.11,
                    subplot_titles=('Typical error — MdAPE % (lower better)', 'Hit-rate — within ±20% (higher better)'))
for nm, e, w, c in [('Proforma v1.0', 'pf_e', 'pf_w', C_PF), ('Model v1.5', 'md_e', 'md_w', C_MODEL),
                    ('Model 5', 'm5_e', 'm5_w', C_M5)]:
    fig.add_trace(go.Bar(x=xa, y=A[e], name=nm, marker_color=c, text=A[e].round(0), textposition='outside'), 1, 1)
    fig.add_trace(go.Bar(x=xa, y=A[w], marker_color=c, showlegend=False, text=A[w].round(0), textposition='outside'), 1, 2)
fig.update_layout(height=470, width=1000, template='plotly_white', barmode='group',
                  title=dict(text='Forecast accuracy by operating year', y=0.97),
                  legend=dict(orientation='h', yanchor='top', y=-0.16, xanchor='center', x=0.5), margin=dict(t=70, b=95))
fig.update_yaxes(title_text='MdAPE %', row=1, col=1); fig.update_yaxes(title_text='within ±20% (%)', row=1, col=2)
fig.show()
print(A[['yr', 'n', 'pf_e', 'md_e', 'm5_e', 'pf_r', 'md_r', 'm5_r']].round(2).to_string(index=False))""")

md("""**Insights**
- **Model 5 is the most accurate forecast in almost every year** (MdAPE e.g. Y4 **19%** vs 31/32%,
  Y5 25% vs 38/35%, Y1 41% vs 49/51%), and lifts the hit-rate too. Feeding the site's own
  pay/vacuum/type turns the count from a coin-toss into the best guess available.
- **Proforma v1.0 and v1.5 are ~tied on yearly error** (49/40/39 vs 51/42/36) — the honest reason:
  a *location-only* model and an *expert-curated linear formula* both hit the same ~±40% ceiling of
  early-year noise. Neither knows the site's hardware; Model 5 does, and that's the whole difference.
- (Ranking check, for site *selection* rather than the count: Model 5 and v1.5 both order sites far
  better than the proforma — Spearman ρ ≈ 0.55–0.60 vs 0.25–0.38 — printed above.)""")

md("""## 3. Scorecard — the three forecasters side by side

Pooled over all site-years and at maturity, worst → best. **Model 5** is the shipped super
ensemble — v1.5's plateau recalibrated with the site's pay-station / vacuum / type inputs.""")

code("""# pooled site-years, computed live on identical sites for all three forecasters
py = []
for y in range(1, 6):
    d = D[D.open_observed & D[f'nobs_y{y}'].ge(10) & D[f'act_y{y}'].gt(0)
          & D[f'pf_y{y}'].gt(0) & D[f'cs_y{y}'].gt(0) & D[f'm5_y{y}'].gt(0)]
    py.append(d[[f'act_y{y}', f'pf_y{y}', f'cs_y{y}', f'm5_y{y}']]
              .rename(columns={f'act_y{y}': 'a', f'pf_y{y}': 'pf', f'cs_y{y}': 'cs', f'm5_y{y}': 'm5'}))
P = pd.concat(py, ignore_index=True)
dm = MATURE.dropna(subset=['pf_y5', 'plateau_loo', 'm5_mature'])
def scard(name, sy, mat):
    return dict(forecaster=name,
        site_yr_MdAPE=f'{mdape(P.a, sy):.0f}%', site_yr_within20=f'{within(P.a, sy):.0f}%',
        mature_MdAPE=f'{mdape(dm.mature_wash, mat):.0f}%', mature_within20=f'{within(dm.mature_wash, mat):.0f}%',
        mature_rho=f'{rho(dm.mature_wash, mat):.2f}', mature_bias=f'{ratio_med(dm.mature_wash, mat):.2f}')
tbl = pd.DataFrame([
    scard('Proforma v1.0', P.pf, dm.pf_y5),
    scard('Model v1.5 (location only)', P.cs, dm.plateau_loo),
    scard('Model 5 (＋ site factors)', P.m5, dm.m5_mature),
])
from IPython.display import display
display(tbl)
e_m5 = np.abs(np.log(P.a / P.m5)); e_pf = np.abs(np.log(P.a / P.pf))
print(f'Model 5 beats the old proforma on {(e_m5 < e_pf).mean()*100:.0f}% of site-years '
      f'(sign test p={stats.binomtest(int((e_m5 < e_pf).sum()), len(P), 0.5).pvalue:.4f})')""")

md("""**Insights**
- **Clean worst → best on mature error: Proforma v1.0 58% → Model v1.5 46% → Model 5 30%** — and
  Model 5 is the only one that is also unbiased (bias 1.00 vs 0.71/0.73) and the only one that lifts
  the within-±20% hit-rate meaningfully (≈39% vs ~21–24%).
- **Model 5 wins the count on ~60%+ of site-years vs the proforma** (sign test printed) — the site
  factors, modelled properly, turn a tie into a decisive win.
- Caveat: Model 5's factor calibration is trained on these 70 mature sites (leave-one-out, so
  honest, but small n) and the pay/vacuum/type inputs exist only where a proforma recorded them.
  Even so, it lands only ~39% within ±20% — quote a range, and remember the ~30% ex-ante ceiling.""")

md("""## 4. Which of the proforma's factors actually predict reality?

The proforma scores ~10 site factors and weights them **roughly equally**. Do they all matter?
For each factor we correlate its score with **actual** mature washes (Spearman), get a
permutation p-value, and apply Benjamini–Hochberg FDR across all factors tested. **Green = real
signal (survives FDR); grey = no significant signal.**""")

code("""def perm_p(x, y, n=20000):
    ok = x.notna() & y.notna(); x, y = x[ok].to_numpy(), y[ok].to_numpy()
    if len(x) < 10 or np.std(x) == 0: return np.nan, np.nan
    xr, yr = stats.rankdata(x), stats.rankdata(y)
    xr = (xr-xr.mean())/xr.std(); yr = (yr-yr.mean())/yr.std(); r0 = float(np.mean(xr*yr))
    perm = np.array([np.mean(xr*rng.permutation(yr)) for _ in range(n)])
    return r0, max((np.abs(perm) >= abs(r0)).mean(), 1/n)
def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); m = ~np.isnan(p); pv = p[m]; k = len(pv)
    o = np.argsort(pv); r = np.minimum.accumulate((pv[o]*k/(np.arange(k)+1))[::-1])[::-1]
    out = np.empty(k); out[o] = np.minimum(r, 1); q[m] = out; return q

DRIVERS = [c for c in FAC if D[c].std() > 1e-9] + ['traffic_count']   # weekly_hours is constant -> excluded
ylog = np.log(MATURE.mature_wash)
res = [dict(driver=c, **dict(zip(['r', 'p'], perm_p(MATURE[c], ylog)))) for c in DRIVERS]
R = pd.DataFrame(res); R['q'] = bh(R.p.values)
R['name'] = (R.driver.str.replace('factor_', '').str.replace('_score', '')
             .str.replace('demog_', '').str.replace('_value', '').str.replace('_', ' '))
R = R.sort_values('r')
fig = go.Figure(go.Bar(x=R.r, y=R.name, orientation='h',
    marker_color=[C_SIG if q < .05 else C_NS for q in R.q],
    text=[f"r={r:.2f}{'  ✓FDR' if q < .05 else ''}" for r, q in zip(R.r, R.q)], textposition='outside'))
fig.update_layout(height=470, width=860, template='plotly_white',
    title='Correlation of each proforma factor with ACTUAL mature washes (n=%d)' % len(MATURE),
    xaxis_title='Spearman r  (green = survives FDR q<0.05)', margin=dict(l=170, t=60))
fig.add_vline(x=0, line_width=1, line_color='#999')
fig.show()
print(R[R.q < .05][['name', 'r', 'q']].round(3).to_string(index=False))""")

md("""**Insights**
- **Exactly the three capacity/format factors survive FDR — pay stations (r=0.41), free vacuum
  slots (0.34), type of site (0.33) — plus the composite score they drive (0.32).** Everything
  else is grey: competition, visibility, accessibility, traffic-speed, area profile and all four
  demographics show **no significant** correlation with real volume.
- **Even the raw traffic count fails (r=0.21, not significant)** — striking, since the proforma
  multiplies its whole forecast by traffic. What matters is the site's *throughput hardware*
  (pay stations, vacuums) and *access* (corner + light), not the demographic profile the template
  spends half its scorecard on.
- So the proforma's equal weighting is wrong: **~half the factors are dead weight.** Caveat: n=70
  and these are correlations — a well-run operator may both build more pay stations and drive more
  volume; the factors screen sites, they don't prove causation.""")

md("""## 5. The factors that matter — what each choice is worth

For the three that survived, the actual mature washes/month behind each ticked box. Monotone,
large steps = a real lever.""")

code("""def norm(v):
    if pd.isna(v): return np.nan
    return str(v).upper().strip().replace('MUTIPLE', 'MULTIPLE').replace('2.0', '2')
LADDERS = [('factor_pay_stations_choice', 'Pay stations', ['1', '2', '3 OR MORE']),
           ('factor_free_vacuum_slots_choice', 'Free vacuum slots',
            ['LESS THAN 12 VEHICLES', '12 - 20 VEHICLES', 'MORE THAN 20 VEHICLES']),
           ('factor_type_of_site_choice', 'Site type', ['INSIDE LOT NO LIGHT', 'CORNER LOT WITHOUT LIGHT',
                                                        'INSIDE LOT NEAR LIGHT', 'CORNER LOT WITH LIGHT'])]
fig = make_subplots(rows=1, cols=3, horizontal_spacing=.07, subplot_titles=[t for _, t, _ in LADDERS])
for col, (cc, title, order) in enumerate(LADDERS, 1):
    m = MATURE.assign(ch=MATURE[cc].map(norm))
    g = m.groupby('ch').mature_wash.agg(['median', 'count'])
    g = g.reindex([o for o in order if o in g.index])
    g = g[g['count'] >= 3]
    labs = [o.title().replace('Or', 'or').replace(' Vehicles', '').replace(' Lot', '<br>Lot') for o in g.index]
    fig.add_trace(go.Bar(x=labs, y=g['median'], marker_color=C_SIG, showlegend=False,
        text=[f'{v:,.0f}<br>(n={int(n)})' for v, n in zip(g['median'], g['count'])],
        textposition='outside'), 1, col)
    fig.update_xaxes(tickangle=-25, row=1, col=col)
fig.update_yaxes(title_text='actual mature washes/mo', row=1, col=1)
fig.update_layout(height=460, width=1020, template='plotly_white',
                  title='What each ticked box is actually worth (median mature washes/mo)', margin=dict(t=90, b=90))
fig.show()""")

md("""**Insights**
- **Every ladder climbs, and the steps are big.** Pay stations 1 → 2 → 3+ ≈ **3.3k → 6.1k → 9.4k**
  washes/mo; vacuums <12 → 12–20 → >20 ≈ **3.6k → 6.3k → 9.1k**. Each rung is roughly +50% volume —
  these are genuine capacity levers, not scorecard decoration.
- **The traffic light beats the corner**: corner+light 9.8k > inside+light 5.9k > corner-no-light
  4.8k > inside-no-light 3.3k — a signal within *both* lot types. The template ranks
  corner-without-light above inside-near-light; the data says that's backwards.
- Caveat: the top rungs (3+ pay stations, >20 vacuums) are small cells (n≈6) — the **direction** is
  solid and monotone, but treat the exact top-end magnitude as indicative, not precise.""")

md("""## 6. Bias vs spread — how *wide* is each forecaster?

The medians hide the spread. Each row below is the full **actual ÷ forecast** ratio for a
forecaster: the **box is the middle 50% of sites**, the line inside is the median, whiskers reach
the tails. Centred on the dashed 1.0 line = unbiased; a narrow box = precise.""")

code('''dm = MATURE.dropna(subset=['pf_y5', 'plateau_loo', 'm5_mature']).copy()
ratios = [('Proforma v1.0', dm.mature_wash / dm.pf_y5, C_PF),
          ('Model v1.5', dm.mature_wash / dm.plateau_loo, C_MODEL),
          ('Model 5', dm.mature_wash / dm.m5_mature, C_M5)]
fig = go.Figure()
for nm, r, c in ratios:                                   # order top->bottom = reverse of add order
    fig.add_trace(go.Box(x=r.clip(upper=3.0), name=nm, marker_color=c, line_color=c,
                         boxpoints='outliers', orientation='h'))
fig.add_vline(x=1.0, line=dict(color=C_ACT, dash='dash'), annotation_text='perfect (unbiased)')
fig.update_layout(height=360, width=880, template='plotly_white', showlegend=False,
    title='Actual / forecast mature washes — bias & spread per forecaster (n=%d)' % len(dm),
    xaxis_title='actual / forecast   (1.0 = perfect;  box = middle 50% of sites)', xaxis_range=[0, 3])
fig.show()
for nm, r, _ in ratios:
    print(f'{nm:14s}: median {r.median():.2f}  IQR [{r.quantile(.25):.2f}, {r.quantile(.75):.2f}]  '
          f'within +-20% {np.mean(np.abs(r-1)<=.2)*100:.0f}%  worst-tenth does {r.quantile(.1):.2f}x forecast')''')

md("""**Insights**
- **Model 5's box straddles the perfect line** (median 1.00) — the factor calibration removes the
  ~0.72 over-projection that both v1.0 and v1.5 share on these older sites, and it lands the most
  sites within ±20% (**39% vs ~21–24%**) with a better low tail (worst-tenth 0.50× vs 0.19×). It
  trades a slightly fatter *high* tail (it over-corrects a few small sites) for being the only
  unbiased forecaster.
- **Even so, precision has a floor:** the raw forecasters land ~21–24% within ±20% and Model 5 ~39% —
  none is a point instrument. That's the concrete answer to "the median looks fine, so is it
  accurate?" — the *centre* can be right while any *single* site is still a wide bet, so quote a
  range, don't promise a number.
- The shared ~0.72 over-projection of v1.0/v1.5 here is a **cohort effect** — these matched sites are
  older and weaker than average. §7 shows v1.5 is essentially unbiased on the full 862-site base.""")

md("""## 7. Reality check at scale — 862 sites, not 70

The comparison so far is on the 112 proforma-matched sites (older, weaker than average). Our model
runs the same leave-one-out forecast on **every** eligible panel site (opened 2021–24, ≥24 months
of data). No proforma exists for these, so it's a model-only check — but it answers "is v1.5's
result a small-sample fluke?".""")

code('''pr = json.load(open(f'{BASE}/ensemble/results/panel_results.json'))
yrs = pd.DataFrame(pr['raw_years'])
xa = [f"{r.yr}<br><span style='font-size:10px;color:#999'>n={int(r.n)}</span>" for r in yrs.itertuples()]
fig = go.Figure(go.Bar(x=xa, y=yrs.MdAPE, marker_color=C_MODEL,
    text=[f'{v:.0f}%<br>×{rr:.2f}' for v, rr in zip(yrs.MdAPE, yrs.ratio_med)], textposition='outside'))
fig.update_layout(height=430, width=760, template='plotly_white',
    title='Model v1.5 on 862 panel sites — MdAPE by operating year  (×n.nn = median bias)',
    yaxis_title='MdAPE %', yaxis_range=[0, 44], margin=dict(t=60))
fig.show()
print(f"mature (n={pr['n']}): MdAPE {pr['raw_mature']['MdAPE']:.0f}%  bias {pr['raw_mature']['ratio']:.2f}  "
      f"rho {pr['raw_mature']['rho']:.2f}  within +-20% {pr['raw_mature']['within20']:.0f}%")''')

md("""**Insights**
- **At scale v1.5 is unbiased and steady**: every operating year sits at **~28–34% MdAPE with a bias
  of ≈1.0** (0.99–1.05), and mature bias is **0.99** — the ~0.72 over-projection in §6 was the
  matched cohort, not the model.
- **So the honest accuracy of our forecaster on a representative new site is ~32% typical error with
  no directional bias.** That matches the published ceiling for brand-new-site forecasting (nobody
  beats ~30% ex-ante) — the rest is information the world doesn't hold yet, which §8 shows the site
  itself supplies once it opens.
- Caveat: these 862 are survivors (built, ≥24 months reported); a site that failed fast or never
  opened isn't here, so spread on an unscreened pipeline is a touch wider.""")

md("""## 8. It gets much better after opening

Ex-ante is a range; the site's own numbers sharpen it fast. After opening we blend the model's
prior with the site's first *k* observed months (de-ramped to a mature-equivalent level, with the
blend shrunk so one noisy month can't whipsaw it). The curve is the mature-level error as those
months arrive.""")

code('''cur = pd.DataFrame(json.load(open(f'{BASE}/ensemble/results/postopen_curve.json'))['curve'])
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0] + list(cur.k), y=[cur.prior_mdape.iloc[0]] + list(cur.post_mdape),
    mode='lines+markers', line=dict(color=C_MODEL, width=3), marker=dict(size=9),
    name='forecast (prior + observed months)'))
fig.add_hline(y=float(cur.prior_mdape.mean()), line=dict(color=C_ACT, dash='dash'),
    annotation_text='ex-ante prior — if you never re-forecast', annotation_position='top right')
fig.update_layout(height=430, width=820, template='plotly_white', showlegend=False,
    title='Mature-level error vs months of actuals observed (862 sites)',
    xaxis_title='months of real operation observed', yaxis_title='MdAPE %', yaxis_range=[0, 38], margin=dict(t=60))
fig.show()
print(cur[['k', 'n', 'tau', 'post_mdape', 'post_w20']].round(0).to_string(index=False))''')

md("""**Insights**
- **Six real months roughly halves the gap to perfect**: mature MdAPE falls from **~33% ex-ante to
  ~25% at 6 months and ~20% by 12**, while the within-±20% hit-rate climbs **33% → 44% → 49%**. A
  site's own ramp is worth more than any amount of pre-build modelling.
- **Product rule:** forecast a *range* at pin-drop, then **re-forecast monthly after opening** — by
  a site's first anniversary you're quoting a genuinely tight number, not a prior.
- The blend is deliberately shrunk (it weights the prior like ~4–6 months of data, fitted
  out-of-fold): at k=1 it barely moves off the prior, then tightens as evidence accumulates — so
  the curve is smooth, not jumpy.""")

md("""## Findings

1. **A linear formula only ties *location-only* modelling.** Per operating year, Proforma v1.0 and
   Model v1.5 are tied at ~±40% error — because neither knows the site's hardware, so both hit the
   early-year noise ceiling. It is not a fair fight for "proper modelling" until the model gets the
   site's own inputs.
2. **Model 5 breaks the tie decisively.** Feeding the three proven factors (pay stations, vacuums,
   site type — from the proforma's own data) into the model cuts mature error to **30% vs v1.5's 46%
   and the proforma's 58%**, makes it **unbiased** (bias 1.00), and wins the count on ~60%+ of
   site-years. Proper modelling *does* win — once it uses the inputs that matter.
3. **Only three factors matter.** Pay stations, free-vacuum slots and site type (corner + light)
   survive FDR against real volume; competition, visibility, accessibility, traffic-speed, area
   profile, all demographics — and even raw traffic count — do not. The proforma's equal weighting
   spends half its scorecard on noise; Model 5 keeps only the signal.
4. **Still not a point instrument.** Even Model 5 lands only ~39% of mature sites within ±20% (the
   raw forecasters ~21–24%); at scale v1.5 is unbiased at **~32% typical error** — the published
   ceiling for brand-new-site forecasting. Rank sites and quote a range, don't promise a number.
5. **And it sharpens fast after opening.** Blending the prior with a site's own first 6 months cuts
   mature error from ~33% to ~25% (to ~20% by 12 months). The workflow: score on capacity +
   signalized access, forecast with Model 5, quote a range at pin-drop, then re-forecast monthly
   once the site is live.""")

nb['cells'] = cells
out = os.path.join(os.path.dirname(__file__), '..', 'proforma_backtest.ipynb')
nbf.write(nb, out)
print('wrote', out, 'with', len(cells), 'cells')
