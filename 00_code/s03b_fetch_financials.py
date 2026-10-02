import time
import pandas as pd
from config import META, sec_session, sec_get, RATE_SLEEP

FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{:010d}.json"

WANTED = {
    "revenue":   ["RevenueFromContractWithCustomerExcludingAssessedTax",
                  "RevenueFromContractWithCustomerIncludingAssessedTax",
                  "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet",
                  "SalesRevenueServicesNet", "RevenuesNetOfInterestExpense"],
    "assets":    ["Assets"],
    "equity":    ["StockholdersEquity"],
    "liab":      ["Liabilities"],
    "netincome": ["NetIncomeLoss"],
    "ppe":       ["PropertyPlantAndEquipmentNet"],
    "rd":        ["ResearchAndDevelopmentExpense"],
    "employees": ["EntityNumberOfEmployees"],
}

def extract(facts, tags):
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei  = facts.get("facts", {}).get("dei", {})
    merged = {}
    for tag in reversed(tags):
        node = gaap.get(tag) or dei.get(tag)
        if not node:
            continue
        best = {}
        for unit_rows in node.get("units", {}).values():
            for row in unit_rows:
                if row.get("fp") == "FY" and row.get("form", "").startswith("10-K") and row.get("fy"):
                    key = int(row["fy"])
                    if key not in best or row.get("filed", "") >= best[key][1]:
                        best[key] = (row.get("val"), row.get("filed", ""))
        merged.update({k: v[0] for k, v in best.items()})
    return merged

def main():
    uni  = pd.read_csv(META / "industries_10k_universe.csv")
    firms = uni[["cik", "name"]].drop_duplicates("cik")
    s, rows = sec_session(), []

    for n, (_, f) in enumerate(firms.iterrows(), 1):
        cik = int(f.cik)
        try:
            facts = sec_get(s, FACTS.format(cik)).json()
            series = {k: extract(facts, tags) for k, tags in WANTED.items()}
            years  = set().union(*[set(v) for v in series.values()]) if series else set()
            for y in sorted(years):
                if 2014 <= y <= 2025:
                    rows.append({"cik": cik, "name": f["name"], "fy": y,
                                 **{k: series[k].get(y) for k in WANTED}})
        except Exception as e:
            print(f"  [{n:>3}/{len(firms)}] {f['name'][:36]:<36} FAILED {type(e).__name__}", flush=True)
        if n % 25 == 0:
            print(f"  [{n:>3}/{len(firms)}] ...", flush=True)
        time.sleep(RATE_SLEEP)

    fin = pd.DataFrame(rows)
    fin.to_csv(META / "firm_financials.csv", index=False)
    print(f"\nFinancial rows: {len(fin):,} firm-years for {fin.cik.nunique()} firms")
    print(f"Revenue coverage: {fin.revenue.notna().mean():.1%} | Assets: {fin.assets.notna().mean():.1%}")
    print(f"Saved -> {META / 'firm_financials.csv'}")

if __name__ == "__main__":
    main()
