# 6-Year Site KPI Explorer

Self-contained Streamlit app for exploring 6 years of per-site KPIs.

## Contents

```
six_year_kpi_explorer/
├── app.py                              # the Streamlit app
├── requirements.txt                    # pinned Python dependencies
├── README.md
└── data/
    └── main-ds.csv                     # KPIs + lat/lon in one file
```

## Setup & run

```bash
# from inside this folder
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app reads `data/main-ds.csv` using a **relative path**, so always launch
it from inside this folder.
