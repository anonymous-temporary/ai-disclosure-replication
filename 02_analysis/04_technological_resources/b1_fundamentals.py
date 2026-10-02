import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import numpy as np
import pandas as pd
import core

def winsor(s, by):
    def w(x):
        if x.notna().sum() < 30: return x
        return x.clip(x.quantile(.01), x.quantile(.99))
    return s.groupby(by).transform(w)

def main():
    core.ensure_dirs()
    uni = core.load_universe()[["adsh", "cik", "fy", "period", "industry", "sic"]].copy()
    uni["period_dt"] = pd.to_datetime(uni.period.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

    cc = pd.read_parquet(core.WRDS / "comp_company.parquet")[["gvkey", "cik", "naics"]]
    cc["cik"] = pd.to_numeric(cc.cik, errors="coerce"); cc = cc.dropna(subset=["cik"]); cc["cik"] = cc.cik.astype(int)
    fa = pd.read_parquet(core.WRDS / "comp_funda.parquet").drop(columns=["cik"], errors="ignore").merge(cc[["gvkey", "cik", "naics"]], on="gvkey")
    fa["datadate"] = pd.to_datetime(fa.datadate)
    m = uni.merge(fa, on="cik", how="left", suffixes=("", "_cs"))
    m["gap"] = (m.datadate - m.period_dt).dt.days.abs()
    m = m[m.gap <= 10].sort_values(["adsh", "gap"]).drop_duplicates("adsh")
    cs = m[["adsh", "gvkey", "naics", "datadate", "sale", "at", "xrd", "oiadp", "ni", "emp", "capx", "che", "dltt", "dlc", "lt", "ceq",
            "csho", "prcc_f", "mkvalt", "xsga"]].copy()
    num = [c for c in cs.columns if c not in ("adsh", "gvkey", "naics", "datadate")]
    cs[num] = cs[num].apply(pd.to_numeric, errors="coerce").astype("float64")
    for c in ("sale", "at", "xrd", "oiadp", "ni", "capx", "che", "dltt", "dlc", "lt", "ceq", "mkvalt", "xsga"):
        cs[c] = cs[c] * 1e6
    cs["mktcap"] = cs.mkvalt.fillna(cs.csho * 1e6 * cs.prcc_f)
    cs["emp"] = cs.emp * 1e3

    x1 = pd.read_csv(core.META / "firm_financials.csv"); x2 = pd.read_csv(core.RAW / "sec_xbrl_extra" / "firm_financials_extra.csv")
    xb = x1.merge(x2.drop(columns=["name"]), on=["cik", "fy"], how="outer")
    d = uni.merge(cs, on="adsh", how="left").merge(xb, on=["cik", "fy"], how="left", suffixes=("", "_x"))
    pick = lambda a, b: d[a].where(d[a].notna(), d[b])
    d["src"] = np.where(d["at"].notna(), "compustat", np.where(d.assets.notna(), "xbrl", "none"))
    d["REV"] = pick("sale", "revenue"); d["AT"] = pick("at", "assets"); d["RD"] = pick("xrd", "rd")
    d["OPINC"] = pick("oiadp", "opincome"); d["NI"] = pick("ni", "netincome"); d["CAPX"] = pick("capx", "capex")
    d["CASH"] = pick("che", "cash"); d["EMP"] = pick("emp", "employees")
    debt_cs = d.dltt.fillna(0) + d.dlc.fillna(0); debt_x = d.ltdebt.fillna(0) + d.debt_cur.fillna(0)
    d["DEBT"] = np.where(d.dltt.notna() | d.dlc.notna(), debt_cs, np.where(d.ltdebt.notna() | d.debt_cur.notna(), debt_x, np.nan))
    d["MKTCAP"] = d.mktcap.where(d.mktcap.notna(), d.public_float)
    d["BOOK"] = d.ceq.where(d.ceq.notna(), d.equity)
    for c in ("REV", "AT", "MKTCAP"):
        d.loc[d[c] <= 0, c] = np.nan
    capx_x = d.capx.isna() & d.capex.notna()
    d.loc[capx_x, "CAPX"] = d.loc[capx_x, "CAPX"].abs(); d.loc[d.CAPX < 0, "CAPX"] = 0.0
    d.loc[d.RD < 0, "RD"] = 0.0
    d.loc[d.DEBT < 0, "DEBT"] = np.nan; d.loc[d.CASH < 0, "CASH"] = np.nan
    d = d.sort_values(["cik", "fy"]).reset_index(drop=True)

    d["RD_MISSING"] = d.RD.isna().astype(int)
    d["PRE_REVENUE"] = (d.REV.fillna(0) < 1e6).astype(int)
    d["RD_SALES_raw"] = d.RD / d.REV
    d["RD_SALES"] = d.RD_SALES_raw.clip(upper=1.0)
    d.loc[(d.RD_MISSING == 0) & (d.PRE_REVENUE == 1), "RD_SALES"] = 1.0
    d["RD_SALES0"] = d.RD_SALES.fillna(0)
    d["RD_AT"] = d.RD / d.AT
    d["LOG_RD"] = np.log1p(d.RD.fillna(0) / 1e6)
    d["RD_MKTCAP"] = d.RD / d.MKTCAP
    stock = []
    for _, g in d.groupby("cik", sort=False):
        k, prev = 0.0, None
        for fy, rd in zip(g.fy, g.RD.fillna(0)):
            k = k * (0.85 ** (1 if prev is None else max(1, int(fy - prev)))) + rd; prev = fy; stock.append(k)
    d["RD_STOCK_AT"] = np.array(stock) / d.AT

    g = d.groupby("cik")
    adj = lambda h: g.fy.shift(-h) == d.fy + h
    d["OP_MARGIN"] = (d.OPINC / d.REV).where(d.PRE_REVENUE == 0)
    lagAT = g.AT.shift(1).where(g.fy.shift(1) == d.fy - 1)
    d["ROA"] = d.NI / ((d.AT + lagAT) / 2).fillna(d.AT)
    for h in (1, 2, 3):
        d[f"REV_GROWTH_{h}"] = np.log(g.REV.shift(-h).where(adj(h)) / d.REV).where(d.PRE_REVENUE == 0)
        d[f"D_OP_MARGIN_{h}"] = d.groupby("cik").OP_MARGIN.shift(-h).where(adj(h)) - d.OP_MARGIN
        d[f"D_ROA_{h}"] = d.groupby("cik").ROA.shift(-h).where(adj(h)) - d.ROA
    d["SIZE"] = np.log(d.AT); d["LEV"] = d.DEBT / d.AT; d["CASH_AT"] = d.CASH / d.AT; d["CAPEX_AT"] = d.CAPX / d.AT
    d["LOG_EMP"] = np.log1p(d.EMP); d["LOG_MKTCAP"] = np.log(d.MKTCAP); d["BTM"] = d.BOOK / d.MKTCAP
    d["LOSS"] = (d.NI < 0).fillna(False).astype(int)

    ratios = ["RD_SALES_raw", "RD_AT", "RD_MKTCAP", "RD_STOCK_AT", "OP_MARGIN", "ROA", "LEV", "CASH_AT", "CAPEX_AT", "BTM"] + \
             [f"{v}_{h}" for v in ("REV_GROWTH", "D_OP_MARGIN", "D_ROA") for h in (1, 2, 3)]
    for c in ratios:
        d[c] = d[c].replace([np.inf, -np.inf], np.nan); d[c + "_w"] = winsor(d[c], d.fy)
    prev_ok = d.groupby("cik").fy.shift(1) == d.fy - 1
    for c in ["RD_SALES", "RD_SALES0", "RD_MISSING", "PRE_REVENUE", "RD_AT_w", "RD_STOCK_AT_w", "LOG_RD", "RD_MKTCAP_w", "SIZE", "LEV_w",
              "CASH_AT_w", "CAPEX_AT_w", "ROA_w", "OP_MARGIN_w", "LOSS", "LOG_MKTCAP", "BTM_w"]:
        d["L1_" + c] = d.groupby("cik")[c].shift(1).where(prev_ok)
    keep = [c for c in d.columns if c.isupper() or c.startswith("L1_") or c.endswith("_w") or c in ("adsh", "cik", "fy", "industry", "sic", "gvkey", "naics", "src", "datadate")]
    d[keep].to_csv(core.out("fundamentals_firm_year.csv"), index=False)

    print(f"{len(d):,} firm-years | source: {d.src.value_counts().to_dict()}")
    cov = d[["RD_SALES", "RD_AT", "RD_MKTCAP", "OP_MARGIN", "ROA", "REV_GROWTH_1", "REV_GROWTH_2", "REV_GROWTH_3", "LEV", "CASH_AT", "CAPEX_AT", "LOG_EMP", "LOG_MKTCAP"]].notna().mean()
    print("coverage:\n" + cov.round(3).to_string())
    print("\nR&D / revenue (capped at 1) by industry: share reporting R&D, median among reporters")
    print(d.groupby("industry").agg(reports_rd=("RD_MISSING", lambda s: 1 - s.mean()), rd_sales_median=("RD_SALES", "median"),
                                    pre_revenue=("PRE_REVENUE", "mean")).round(3).to_string())

if __name__ == "__main__":
    main()
