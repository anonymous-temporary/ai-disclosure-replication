import time
import pandas as pd
from config import EXT, META, sec_session, sec_get, RATE_SLEEP

OUT = EXT / "sec_xbrl_extra"
OUT.mkdir(parents=True, exist_ok=True)

FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{:010d}.json"

WANTED = {
    "opincome":   ["OperatingIncomeLoss"],
    "cogs":       ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
                   "CostOfServices"],
    "sga":        ["SellingGeneralAndAdministrativeExpense"],
    "opex":       ["OperatingExpenses", "CostsAndExpenses"],
    "cash":       ["CashAndCashEquivalentsAtCarryingValue",
                   "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "ltdebt":     ["LongTermDebtNoncurrent", "LongTermDebt"],
    "debt_cur":   ["LongTermDebtCurrent", "DebtCurrent"],
    "capex":      ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "intangibles":["IntangibleAssetsNetExcludingGoodwill"],
    "goodwill":   ["Goodwill"],
    "shares_out": ["EntityCommonStockSharesOutstanding",
                   "CommonStockSharesOutstanding"],
    "public_float":["EntityPublicFloat"],
    "eps_diluted":["EarningsPerShareDiluted"],
}

def extract(facts, tags):
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei  = facts.get("facts", {}).get("dei", {})
    merged, ends = {}, {}
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
                        best[key] = (row.get("val"), row.get("filed", ""), row.get("end"))
        merged.update({k: v[0] for k, v in best.items()})
        ends.update({k: v[2] for k, v in best.items()})
    return merged, ends

def main():
    uni   = pd.read_csv(META / "industries_10k_universe.csv")
    firms = uni[["cik", "name"]].drop_duplicates("cik")
    s, rows = sec_session(), []

    for n, (_, f) in enumerate(firms.iterrows(), 1):
        cik = int(f.cik)
        try:
            facts  = sec_get(s, FACTS.format(cik)).json()
            series, fyend = {}, {}
            for k, tags in WANTED.items():
                series[k], e = extract(facts, tags)
                for y, d in e.items():
                    if d and (y not in fyend or k == "opincome"):
                        fyend[y] = d
            years = set().union(*[set(v) for v in series.values()]) if series else set()
            for y in sorted(years):
                if 2012 <= y <= 2025:
                    rows.append({"cik": cik, "name": f["name"], "fy": y,
                                 "fy_end": fyend.get(y),
                                 **{k: series[k].get(y) for k in WANTED}})
        except Exception as e:
            print(f"  [{n:>4}/{len(firms)}] {f['name'][:36]:<36} FAILED {type(e).__name__}", flush=True)
        if n % 50 == 0:
            print(f"  [{n:>4}/{len(firms)}] ...", flush=True)
        time.sleep(RATE_SLEEP)

    fin = pd.DataFrame(rows)
    fin.to_csv(OUT / "firm_financials_extra.csv", index=False)
    print(f"\nRows: {len(fin):,} firm-years for {fin.cik.nunique()} firms")
    for k in WANTED:
        print(f"  {k:<13} coverage {fin[k].notna().mean():.1%}")
    print(f"Saved -> {OUT / 'firm_financials_extra.csv'}")

if __name__ == "__main__":
    main()
