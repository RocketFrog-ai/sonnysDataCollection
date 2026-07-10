# Sonny's site-selection / proforma platform

Drop a pin on a US map for a car wash that doesn't exist yet. Get a 5-year monthly wash-count
forecast, a P&L, and what it does to the neighbours. Everything orbits one panel of ~2,100 real
sites (`client_id + site_id`, monthly, 2020→2027).

**Start here**

| I want to… | Go to |
|---|---|
| understand the system | `docs/ARCHITECTURE.md` |
| run something | `docs/ENVIRONMENTS.md` — **three** conda/venv envs, not interchangeable |
| change the forecast | `proforma/MODELLING.md`, then `proforma/models/coldstart.py` |
| find a dataset | `docs/DATA.md` — everything lives once, under `proforma/data/` |
| know what's already broken | `docs/DIVERGENCES.md` — read before "fixing" anything |

```bash
conda activate sonnys                # the one environment (py3.11)
streamlit run proforma/ui/app.py     # the explorer
python -m app.main                   # the API
./scripts/smoke.sh                   # prove you changed no numbers
```

`proforma/` is all the modelling — one tree, versioned with **git tags**, not folders
(`proforma-v1.5` is the current model). `app/` is the FastAPI backend. `archive/` and
`experiments/` are off the import path — read for history, don't build on them.
