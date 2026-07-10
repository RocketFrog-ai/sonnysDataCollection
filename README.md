# Sonny's site-selection / proforma platform

Drop a pin on a US map for a car wash that doesn't exist yet. Get a 5-year monthly wash-count
forecast, a P&L, and what it does to the neighbours. Everything orbits one panel of ~2,100 real
sites (`client_id + site_id`, monthly, 2020→2027).

**Start here**

| I want to… | Go to |
|---|---|
| understand the system | `docs/ARCHITECTURE.md` |
| run something | `docs/ENVIRONMENTS.md` — **three** conda/venv envs, not interchangeable |
| change the forecast | `proforma/v1_5/MODELLING.md`, then `proforma/v1_5/models/coldstart.py` |
| find a dataset | `docs/DATA.md` — everything lives once, under `proforma/data/` |
| know what's already broken | `docs/DIVERGENCES.md` — read before "fixing" anything |

```bash
streamlit run proforma/v1_5/ui/app.py     # the explorer     (conda proforma311)
uvicorn app.pnl_only:app --port 8010      # P&L API only     (conda sonnysDataCollection)
python -m app.main                        # full backend     (conda sonnysDataCollection)
./scripts/smoke.sh                        # prove you changed no numbers
```

`proforma/` is all modelling, versioned (`v1_5` live, `v1_6` experimental). `app/` is the FastAPI
backend. `archive/` and `experiments/` are frozen — read for history, don't build on them.
