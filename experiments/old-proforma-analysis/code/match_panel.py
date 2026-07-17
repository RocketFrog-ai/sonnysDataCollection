"""
Address-match the proforma combined CSV to the actuals panel and append the
matched site's monthly-derived actuals (washcount, revenue, ASP, membership).

Match logic: normalise the panel street (house# + standardised suffix/direction
words) and test it as a substring of the normalised proforma full-address, then
require the state (preferred: ZIP) to agree.  Ambiguous / no-match rows are kept
with blank actuals and a match_status flag.

Same-address operator handoffs: the panel keeps a rebranded site as TWO
client_ids whose windows overlap at the seam month (each operator reports its
part of that month), so the panel's own stitcher skips them as "concurrent".
Here every match is checked for same-street+ZIP twin records; the actuals are
aggregated over the UNION of all segments (seam months resolved by keeping the
row with the larger wash_count — every kept row stays a real panel row) and
match_operational_start becomes the earliest segment's start.  The twins are
recorded in match_secondary_segments ("client_id:site_id;...") so
build_monthly.py can expand the same union.

actuals_suspect flags matched records whose actuals are reporting artifacts
(avg < 200 washes/month across the record's history — real sites do thousands;
the panel's dead records do ~1-100).

Safe to re-run on its own output: pre-existing match_*/actual_* columns are
dropped before matching.
"""
import os
import re
import pandas as pd

HERE = os.path.dirname(__file__)
PF_CSV = os.path.join(HERE, "..", "old-proforma-combined.csv")
PANEL = os.path.join(HERE, "..", "..", "..", "proforma", "data", "panel", "main-data-v2-stitched.csv")

SUFFIX = {
    "street": "st", "avenue": "ave", "av": "ave", "road": "rd", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy",
    "court": "ct", "place": "pl", "circle": "cir", "trail": "trl", "trailway": "trl",
    "terrace": "ter", "square": "sq", "route": "rt", "expressway": "expy",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "saint": "st",
}


def norm_addr(s):
    if s is None or (isinstance(s, float)):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)          # punctuation -> space
    toks = [SUFFIX.get(t, t) for t in s.split()]
    return " ".join(toks).strip()


def extract_zip(s):
    m = re.findall(r"\b(\d{5})\b", str(s))
    return m[-1] if m else ""


STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN "
             "MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA "
             "WA WV WI WY DC".split())


def extract_state(s):
    toks = re.findall(r"[A-Za-z]{2}", str(s).upper())
    for t in reversed(toks):
        if t in STATES:
            return t
    return ""


SUSPECT_WASHES_PER_MONTH = 200   # below this the record is a reporting artifact


def union_months(frames):
    """Union the monthly rows of same-address segments (priority order).

    A month reported by two segments is the handoff seam — each operator
    reports its part of the month — so keep the row with the larger
    wash_count (the operator who ran most of the month); ties go to the
    earlier frame in `frames`.  Every kept row is a real panel row, so the
    ratio columns (ASP, pct) stay internally consistent.
    """
    both = pd.concat(frames, keys=range(len(frames)), names=["seg_rank"]).reset_index(level=0)
    both = both.sort_values(["year", "month", "wash_count", "seg_rank"],
                            ascending=[True, True, False, True])
    return both.drop_duplicates(["year", "month"]).sort_values(["year", "month"])


def agg_from_months(d):
    """The per-site actual_* aggregates, mirroring the groupby in main()."""
    return {
        "actual_n_months": len(d),
        "actual_first_period": f"{int(d['year'].min())}",
        "actual_wash_count_total": d["wash_count"].sum(),
        "actual_revenue_total": d["revenue"].sum(),
        "actual_mem_wash_count_monthly_avg": d["mem_wash_count"].mean(),
        "actual_ret_wash_count_monthly_avg": d["ret_wash_count"].mean(),
        "actual_wash_count_monthly_avg": d["wash_count"].mean(),
        "actual_mem_revenue_monthly_avg": d["mem_revenue"].mean(),
        "actual_ret_revenue_monthly_avg": d["ret_revenue"].mean(),
        "actual_revenue_monthly_avg": d["revenue"].mean(),
        "actual_ASP_mem_avg": d["ASP_mem"].mean(),
        "actual_ASP_ret_avg": d["ASP_ret"].mean(),
        "actual_mem_revenue_pct_avg": d["mem_revenue_pct"].mean(),
        "actual_mem_count_pct_avg": d["mem_count_pct"].mean(),
    }


def main():
    pf = pd.read_csv(PF_CSV)
    # idempotent re-runs: this script rewrites PF_CSV in place, so drop any
    # match/actual columns a previous run appended before matching afresh.
    stale = [c for c in pf.columns
             if c.startswith(("match_", "actual_")) or c == "actuals_suspect"]
    pf = pf.drop(columns=stale)
    panel = pd.read_csv(PANEL)

    # ---- build per-site actuals summary from the monthly panel ----
    sites = panel.drop_duplicates(["client_id", "site_id"]).copy()
    sites["nstreet"] = sites["address1"].map(norm_addr)
    # zfill: MA/NJ/... zips lose their leading zero in the panel ("1835" for
    # 01835), which silently downgraded correct zip matches to state matches.
    sites["zip5"] = sites["postal_code"].astype(str).str.extract(r"(\d{3,5})")[0].str.zfill(5)

    # aggregate monthly -> per-site actuals (totals, means, latest-12mo)
    panel["wash_count"] = panel["mem_wash_count"].fillna(0) + panel["ret_wash_count"].fillna(0)
    panel["revenue"] = panel["mem_revenue"].fillna(0) + panel["ret_revenue"].fillna(0)
    g = panel.groupby(["client_id", "site_id"])
    agg = pd.DataFrame({
        "actual_n_months": g.size(),
        "actual_first_period": g.apply(lambda d: f"{int(d['year'].min())}", include_groups=False),
        "actual_wash_count_total": g["wash_count"].sum(),
        "actual_revenue_total": g["revenue"].sum(),
        "actual_mem_wash_count_monthly_avg": g["mem_wash_count"].mean(),
        "actual_ret_wash_count_monthly_avg": g["ret_wash_count"].mean(),
        "actual_wash_count_monthly_avg": g["wash_count"].mean(),
        "actual_mem_revenue_monthly_avg": g["mem_revenue"].mean(),
        "actual_ret_revenue_monthly_avg": g["ret_revenue"].mean(),
        "actual_revenue_monthly_avg": g["revenue"].mean(),
        "actual_ASP_mem_avg": g["ASP_mem"].mean(),
        "actual_ASP_ret_avg": g["ASP_ret"].mean(),
        "actual_mem_revenue_pct_avg": g["mem_revenue_pct"].mean(),
        "actual_mem_count_pct_avg": g["mem_count_pct"].mean(),
    }).reset_index()

    site_meta = sites[["client_id", "site_id", "client_name", "address1", "state",
                       "postal_code", "zip5", "nstreet", "lat", "lon",
                       "operational_start"]]
    site_full = site_meta.merge(agg, on=["client_id", "site_id"], how="left")

    # street tokens must appear as a contiguous run inside the proforma address
    # (word-boundary safe; substring matching wrongly hit "w st" inside "w state").
    site_full["stoks"] = site_full["nstreet"].apply(lambda s: s.split())

    def street_in(stoks, atoks):
        n = len(stoks)
        if n < 2:                      # need house# + >=1 word to be meaningful
            return False
        return any(atoks[i:i + n] == stoks for i in range(len(atoks) - n + 1))

    def pick(df):
        # prefer the record with the most actual monthly data
        return df.sort_values("actual_n_months", ascending=False, na_position="last").iloc[0]

    # ---- match each proforma to a site ----
    out_rows = []
    for _, r in pf.iterrows():
        naddr = norm_addr(r["address"])
        atoks = naddr.split()
        pzip = extract_zip(r["address"])
        pstate = extract_state(r["address"])

        mask = site_full["stoks"].apply(lambda st: street_in(st, atoks))
        hits = site_full[mask]

        status, chosen = "no_match", None
        if len(hits):
            zmatch = hits[hits["zip5"] == pzip] if pzip else hits.iloc[0:0]
            smatch = hits[hits["state"] == pstate] if pstate else hits.iloc[0:0]
            if len(zmatch) == 1:
                status, chosen = "matched_zip", zmatch.iloc[0]
            elif len(smatch) == 1:
                status, chosen = "matched_state", smatch.iloc[0]
            elif len(zmatch) > 1:
                status, chosen = "matched_zip_multi", pick(zmatch)
            elif len(smatch) > 1:
                status, chosen = "matched_state_multi", pick(smatch)
            elif len(hits) == 1:
                status, chosen = "matched_street_only", hits.iloc[0]
            else:
                status, chosen = "ambiguous", pick(hits)

        # fallback: same ZIP + same house number uniquely identifies the site
        # (recovers missing directionals, dropped suffixes, street-name typos).
        if chosen is None and pzip:
            pnum = next((t for t in atoks if t.isdigit()), None)
            if pnum:
                same = site_full[(site_full["zip5"] == pzip) &
                                 (site_full["stoks"].apply(lambda st: bool(st) and st[0] == pnum))]
                if len(same) == 1:
                    status, chosen = "matched_fuzzy", same.iloc[0]
                elif len(same) > 1:
                    status, chosen = "matched_fuzzy_multi", pick(same)

        # last resort: proforma ZIP is wrong/typo'd -> require same house number,
        # same STATE, and a shared meaningful street-name word (excludes the
        # coincidental same-house-number-different-state collisions).
        # Only when the address has a SINGLE plausible house number -- multiple
        # numbers (e.g. a "1075 Sunrise Corp." prefix + a real street number)
        # make the state fallback ambiguous and prone to false positives.
        if chosen is None and pstate:
            house_nums = [t for t in atoks if t.isdigit() and t != pzip]
            pnum = house_nums[0] if len(house_nums) == 1 else None
            SKIP = {"n", "s", "e", "w", "ne", "nw", "se", "sw", "st", "ave", "rd",
                    "blvd", "dr", "ln", "hwy", "pkwy", "ct", "pl", "cir", "trl",
                    "us", "state", "hwy", "rt", "expy"}
            pwords = {t for t in atoks if len(t) >= 4 and t not in SKIP}
            if pnum and pwords:
                m = (site_full["stoks"].apply(lambda st: bool(st) and st[0] == pnum)
                     & (site_full["state"] == pstate)
                     & site_full["stoks"].apply(lambda st: bool(pwords & set(st))))
                same = site_full[m]
                if len(same) == 1:
                    status, chosen = "matched_fuzzy_state", same.iloc[0]

        row = r.to_dict()
        row["match_status"] = status
        if chosen is not None:
            row["match_client_name"] = chosen["client_name"]
            row["match_client_id"] = chosen["client_id"]
            row["match_site_id"] = chosen["site_id"]
            row["match_address1"] = chosen["address1"]
            row["match_state"] = chosen["state"]
            row["match_postal_code"] = chosen["postal_code"]
            row["match_lat"] = chosen["lat"]
            row["match_lon"] = chosen["lon"]
            row["match_operational_start"] = chosen["operational_start"]
            for c in agg.columns:
                if c not in ("client_id", "site_id"):
                    row[c] = chosen[c]

            # same-address twin segments (operator handoffs the panel keeps
            # as separate client_ids because their windows share the seam
            # month): aggregate actuals over the union of all segments.
            twins = site_full[
                (site_full["nstreet"] == chosen["nstreet"])
                & (site_full["nstreet"].str.len() > 0)
                & ~((site_full["client_id"] == chosen["client_id"])
                    & (site_full["site_id"] == chosen["site_id"]))
            ]
            if pd.notna(chosen["zip5"]):
                twins = twins[twins["zip5"] == chosen["zip5"]]
            else:
                twins = twins[twins["state"] == chosen["state"]]
            twins = twins.sort_values("actual_n_months", ascending=False)

            segs = [chosen] + [t for _, t in twins.iterrows()]
            row["match_stitched"] = len(segs) > 1
            row["match_secondary_segments"] = ";".join(
                f"{s['client_id']}:{int(s['site_id'])}" for s in segs[1:])
            if len(segs) > 1:
                u = union_months([
                    panel[(panel["client_id"] == s["client_id"])
                          & (panel["site_id"] == s["site_id"])] for s in segs])
                for k, v in agg_from_months(u).items():
                    row[k] = v
                starts = pd.to_datetime([s["operational_start"] for s in segs],
                                        format="%m-%Y", errors="coerce").dropna()
                if len(starts):
                    row["match_operational_start"] = starts.min().strftime("%m-%Y")

            row["actuals_suspect"] = bool(
                row["actual_wash_count_monthly_avg"] < SUSPECT_WASHES_PER_MONTH)
        else:
            row["match_stitched"] = False
            row["match_secondary_segments"] = ""
            row["actuals_suspect"] = None
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out.to_csv(PF_CSV, index=False)

    print("match_status counts:")
    print(out["match_status"].value_counts().to_string())
    print("\nmatched rows:", (out["match_status"].str.startswith("matched")).sum(), "/", len(out))

    st = out[out["match_stitched"] == True]  # noqa: E712  (column may hold None)
    print(f"\nstitched same-address handoffs: {len(st)}")
    for _, r in st.iterrows():
        print(f"  {r['address']}\n    primary {r['match_client_id']}:{int(r['match_site_id'])}"
              f" + [{r['match_secondary_segments']}]"
              f" -> {int(r['actual_n_months'])} mo, opens {r['match_operational_start']}")

    sus = out[out["actuals_suspect"] == True]  # noqa: E712
    print(f"\nactuals_suspect (<{SUSPECT_WASHES_PER_MONTH} washes/mo): {len(sus)}")
    for _, r in sus.iterrows():
        print(f"  {r['match_client_id']}:{int(r['match_site_id'])}"
              f" avg={r['actual_wash_count_monthly_avg']:.1f}/mo"
              f" over {int(r['actual_n_months'])} mo  ({r['address']})")

    print("\nsample matches:")
    cols = ["source_file", "address", "match_status", "match_client_name",
            "match_address1", "actual_wash_count_monthly_avg", "actual_revenue_monthly_avg"]
    print(out[out["match_status"].str.startswith("matched")][cols].head(20).to_string())


if __name__ == "__main__":
    main()
