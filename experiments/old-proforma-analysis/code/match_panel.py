"""
Address-match the proforma combined CSV to the actuals panel and append the
matched site's monthly-derived actuals (washcount, revenue, ASP, membership).

Match logic: normalise the panel street (house# + standardised suffix/direction
words) and test it as a substring of the normalised proforma full-address, then
require the state (preferred: ZIP) to agree.  Ambiguous / no-match rows are kept
with blank actuals and a match_status flag.
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


def main():
    pf = pd.read_csv(PF_CSV)
    panel = pd.read_csv(PANEL)

    # ---- build per-site actuals summary from the monthly panel ----
    sites = panel.drop_duplicates(["client_id", "site_id"]).copy()
    sites["nstreet"] = sites["address1"].map(norm_addr)
    sites["zip5"] = sites["postal_code"].astype(str).str.extract(r"(\d{5})")[0]

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
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out.to_csv(PF_CSV, index=False)

    print("match_status counts:")
    print(out["match_status"].value_counts().to_string())
    print("\nmatched rows:", (out["match_status"].str.startswith("matched")).sum(), "/", len(out))
    print("\nsample matches:")
    cols = ["source_file", "address", "match_status", "match_client_name",
            "match_address1", "actual_wash_count_monthly_avg", "actual_revenue_monthly_avg"]
    print(out[out["match_status"].str.startswith("matched")][cols].head(20).to_string())


if __name__ == "__main__":
    main()
