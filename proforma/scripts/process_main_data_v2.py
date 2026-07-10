"""Process main-data-v2.csv into the legacy main-ds.csv column schema.

Reads the corrected v2 dataset and emits a file whose columns match the old
main-ds.csv so downstream modelling code keeps working. Key derivations:
  * split the `month` date (YYYY-MM-01) into integer `year` + `month` (1-12)
  * ASP columns (revenue / wash count), same formula used in main-ds
  * `operational_start` = first reporting month per unique site (client_id+site_id)

Run:  python proforma/scripts/process_main_data_v2.py   (from the repo root)
"""
import re

import numpy as np
import pandas as pd

# The panel used to be mirrored in two project data dirs and this script wrote both, in place.
# The mirrors were verified byte-identical and collapsed into one shared store, so there is now
# exactly one input and the "keep them in sync" step no longer exists. Paths are repo-root
# relative: run this from the repo root.
INPUTS = [
    "proforma/data/panel/main-data-v2.csv",
]
SUFFIX = "-processed"  # main-data-v2.csv -> main-data-v2-processed.csv

# Sonny's internal demo account (Tamarac FL HQ) — not real sites; dropped entirely.
DEMO_CLIENTS = {"demo_client"}

# Manual handoff merges the automated (lat, lon, street) rule can't catch — e.g. a
# corner lot with two frontage addresses on different streets. Each entry lists the
# (client_id, site_id) fragments at one physical location; the largest-period fragment
# wins the identity, same as the automated path.
MANUAL_HANDOFFS = [
    # 799 Garrisonville Rd / 110 Patriot Crossing Drive — same lat/lon, corner parcel.
    [("patriotscw_000465", 1), ("edgeexpress_000446", 8)],
]

# US Census regions (states not listed -> NaN region, e.g. Canada/AU, matching old data)
_REGION = {
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest":   ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South":     ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV",
                  "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
    "West":      ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY",
                  "AK", "CA", "HI", "OR", "WA"],
}
STATE_TO_REGION = {s: r for r, states in _REGION.items() for s in states}

# columns in the exact order of the legacy main-ds.csv
OUT_COLS = [
    "client_name", "client_id", "site_id", "year", "month",
    "mem_purchase_count", "mem_wash_count", "mem_revenue",
    "ret_wash_count", "ret_revenue", "ASP_mem", "ASP_ret",
    "mem_revenue_pct", "mem_count_pct",
    "timezone", "address1", "state", "postal_code", "region",
    "lat", "lon", "operational_start",
]


def process(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[~df["client_id"].isin(DEMO_CLIENTS)].reset_index(drop=True)
    dt = pd.to_datetime(df["month"])

    out = pd.DataFrame()
    out["client_name"] = df["name"]
    out["client_id"] = df["client_id"]
    out["site_id"] = df["site_id"]

    # 1) split date -> year + month number
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month

    # rename metric columns to legacy names
    out["mem_purchase_count"] = df["total_membership_purchase_count"]
    out["mem_wash_count"] = df["member_car_wash_count"]
    out["mem_revenue"] = df["car_wash_sales_via_members"]
    out["ret_wash_count"] = df["retail_car_wash_count"]
    out["ret_revenue"] = df["retail_net_sales_amount"]

    # 2) ASP: membership = revenue / PURCHASES, retail = revenue / wash count
    #    (NaN when denominator is 0)
    mp = out["mem_purchase_count"].replace(0, np.nan)
    rw = out["ret_wash_count"].replace(0, np.nan)
    out["ASP_mem"] = (out["mem_revenue"] / mp).round(2)
    out["ASP_ret"] = (out["ret_revenue"] / rw).round(2)

    # revenue / count membership share (%), matching main-ds
    tot_rev = out["mem_revenue"] + out["ret_revenue"]
    tot_wash = out["mem_wash_count"] + out["ret_wash_count"]
    out["mem_revenue_pct"] = np.where(tot_rev > 0, out["mem_revenue"] / tot_rev * 100, np.nan).round(2)
    out["mem_count_pct"] = np.where(tot_wash > 0, out["mem_wash_count"] / tot_wash * 100, np.nan).round(2)

    out["timezone"] = df["timezone"]
    out["address1"] = df["address1"]
    out["state"] = df["state"]
    out["postal_code"] = df["postal_code"]
    out["region"] = df["state"].map(STATE_TO_REGION)
    out["lat"] = df["latitude"]
    out["lon"] = df["longitude"]

    # 3) operational_start = first reporting month per unique site, as MM-YYYY
    first = dt.groupby([df["client_id"], df["site_id"]]).transform("min")
    out["operational_start"] = first.dt.strftime("%m-%Y")

    return out[OUT_COLS]


def stitch_handoffs(out: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Merge same-location operator handoffs into one continuous location trajectory.

    A handoff = several distinct sites (client_id+site_id) that share the exact same
    (lat, lon, address1) and whose reporting windows do NOT overlap in time — i.e. one
    site closes and another opens at the same physical car wash. Requiring identical
    address auto-excludes placeholder coords (0,0 / corporate rollups) whose addresses
    differ, and the non-overlap check excludes genuinely concurrent co-located sites.

    Each such chain is collapsed: every fragment's rows are relabeled to the identity
    (client_id, site_id, client_name) of the LARGEST-PERIOD fragment, and the whole
    chain's `operational_start` is set to the earliest reporting month at the location.
    No rows are dropped — the smaller site identities simply cease to exist as separate
    sites and their months are absorbed into the surviving trajectory.
    """
    df = out.copy()
    d = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df["_d"] = d
    df["_site"] = df.client_id.astype(str) + "|" + df.site_id.astype(str)
    # street name = address with the leading house number stripped. Matching on
    # (lat, lon, street) instead of the full address absorbs trivial house-number
    # discrepancies (e.g. "110" vs "101 Market Center Way" = same parcel) while
    # still keeping genuinely different roads at a shared bad geocode apart.
    df["_street"] = (df.address1.astype(str).str.replace(r"^\s*\d+\s*", "", regex=True)
                     .str.lower().str.strip())

    key = ["lat", "lon", "_street"]
    nsite = df.groupby(key)._site.nunique()
    remap = {}                       # old (client_id, site_id) -> winner identity + loc start
    cases = []
    for k in nsite[nsite > 1].index:
        sub = df[(df.lat == k[0]) & (df.lon == k[1]) & (df._street == k[2])]
        spans = [dict(site=s, cid=g.client_id.iloc[0], sid=g.site_id.iloc[0],
                      name=g.client_name.iloc[0], start=g._d.min(), end=g._d.max(),
                      months=g._d.nunique())
                 for s, g in sub.groupby("_site")]
        spans.sort(key=lambda x: x["start"])
        if any(b["start"] <= a["end"] for a, b in zip(spans, spans[1:])):
            continue                 # temporal overlap -> concurrent, not a handoff
        win = max(spans, key=lambda x: (x["months"], x["end"]))   # largest period wins
        loc_start = min(s["start"] for s in spans).strftime("%m-%Y")
        for s in spans:
            remap[(s["cid"], s["sid"])] = (win["cid"], win["sid"], win["name"], loc_start)
        cases.append((k[2], spans, win, loc_start))

    # manual overrides (corner lots etc. the street rule can't see)
    for frags in MANUAL_HANDOFFS:
        sub = df[df.apply(lambda r: (r.client_id, r.site_id) in frags, axis=1)]
        spans = [dict(cid=g.client_id.iloc[0], sid=g.site_id.iloc[0],
                      name=g.client_name.iloc[0], start=g._d.min(), end=g._d.max(),
                      months=g._d.nunique())
                 for _, g in sub.groupby(["client_id", "site_id"])]
        if len(spans) < 2:
            continue
        win = max(spans, key=lambda x: (x["months"], x["end"]))
        loc_start = min(s["start"] for s in spans).strftime("%m-%Y")
        for s in spans:
            remap[(s["cid"], s["sid"])] = (win["cid"], win["sid"], win["name"], loc_start)
        cases.append(("manual", spans, win, loc_start))

    for (cid, sid), (wcid, wsid, wname, loc_start) in remap.items():
        m = (df.client_id == cid) & (df.site_id == sid)
        df.loc[m, ["client_id", "site_id", "client_name", "operational_start"]] = \
            [wcid, wsid, wname, loc_start]

    # Fill the SEAM gap months (between fragments) so each merged trajectory is
    # continuous. Gap rows carry the winner identity and the merged trajectory's
    # AVERAGE for each base metric; derived columns are recomputed from those means.
    # imputed=1 marks them so modelling can distinguish synthetic from reported rows.
    df["imputed"] = 0
    METRICS = ["mem_purchase_count", "mem_wash_count", "mem_revenue",
               "ret_wash_count", "ret_revenue"]
    gap_rows, n_gap = [], 0
    for _label, spans, win, loc_start in cases:
        spans = sorted(spans, key=lambda x: x["start"])
        seam = []
        for a, b in zip(spans, spans[1:]):
            seam += list(pd.period_range(a["end"] + pd.offsets.MonthBegin(1),
                                         b["start"] - pd.offsets.MonthBegin(1), freq="M"))
        if not seam:
            continue
        loc = df[(df.client_id == win["cid"]) & (df.site_id == win["sid"])]
        avg = {c: loc[c].mean() for c in METRICS}
        tmpl = loc.iloc[0].to_dict()
        for p in seam:
            r = dict(tmpl)
            r["year"], r["month"] = p.year, p.month
            r["operational_start"] = loc_start
            r["imputed"] = 1
            for c in METRICS:
                r[c] = round(avg[c]) if "count" in c else round(avg[c], 2)
            r["ASP_mem"] = round(r["mem_revenue"] / r["mem_purchase_count"], 2) if r["mem_purchase_count"] else np.nan
            r["ASP_ret"] = round(r["ret_revenue"] / r["ret_wash_count"], 2) if r["ret_wash_count"] else np.nan
            tot_rev = r["mem_revenue"] + r["ret_revenue"]
            tot_wash = r["mem_wash_count"] + r["ret_wash_count"]
            r["mem_revenue_pct"] = round(r["mem_revenue"] / tot_rev * 100, 2) if tot_rev else np.nan
            r["mem_count_pct"] = round(r["mem_wash_count"] / tot_wash * 100, 2) if tot_wash else np.nan
            gap_rows.append(r)
            n_gap += 1
    if gap_rows:
        df = pd.concat([df, pd.DataFrame(gap_rows)], ignore_index=True)

    if verbose:
        print(f"  stitched {len(cases)} handoff locations "
              f"({len(remap)} sites -> {len({(v[0], v[1]) for v in remap.values()})}), "
              f"{len(remap) - len({(v[0], v[1]) for v in remap.values()})} identities dropped; "
              f"filled {n_gap} seam-gap months (avg-imputed)")

    df = df.drop(columns=["_d", "_site", "_street"])
    df = df.sort_values(["client_id", "site_id", "year", "month"]).reset_index(drop=True)
    return df


def main() -> None:
    for path in INPUTS:
        out = process(path)
        dest = path.replace("main-data-v2.csv", f"main-data-v2{SUFFIX}.csv")
        out.to_csv(dest, index=False)
        print(f"{dest}: {len(out):,} rows, "
              f"{out.groupby(['client_id', 'site_id']).ngroups:,} sites")

        stitched = stitch_handoffs(out, verbose=True)
        sdest = path.replace("main-data-v2.csv", "main-data-v2-stitched.csv")
        stitched.to_csv(sdest, index=False)
        print(f"{sdest}: {len(stitched):,} rows, "
              f"{stitched.groupby(['client_id', 'site_id']).ngroups:,} sites")


if __name__ == "__main__":
    main()
