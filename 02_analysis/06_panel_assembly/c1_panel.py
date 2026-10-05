import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import numpy as np
import pandas as pd
import core


def main():
    F = pd.read_csv(core.out("filings.csv"))[["adsh", "cik", "fy", "name", "industry", "sic", "filed", "n_words", "segmented", "n_ai_terms"]]
    D = pd.read_csv(core.out("disclosure_firm_year.csv")).drop(columns=["cik", "fy", "n_words"])
    m = F.merge(D, on="adsh", how="left")
    fu = pd.read_csv(core.out("fundamentals_firm_year.csv")).drop(columns=["cik", "fy", "industry", "sic"])
    m = m.merge(fu, on="adsh", how="left")

    pat = pd.read_csv(core.out("patents_firm_year.csv"))
    pat["LOG_PAT_STOCK"] = np.log1p(pat.PAT_STOCK); pat["fy"] = pat.year + 1
    pc = ["PAT_STOCK", "LOG_PAT_STOCK", "AI_PAT_STOCK", "LOG_AI_PAT_STOCK", "AI_PAT_SHARE", "PAT_VALUE"]
    m = m.merge(pat[["cik", "fy"] + pc].rename(columns={c: "L1_" + c for c in pc}), on=["cik", "fy"], how="left")
    inwin = m.fy.between(2001, 2024)
    for c in ("L1_PAT_STOCK", "L1_LOG_PAT_STOCK", "L1_AI_PAT_STOCK", "L1_LOG_AI_PAT_STOCK"):
        m.loc[inwin, c] = m.loc[inwin, c].fillna(0)

    bab = pd.read_stata(core.RAW / "babina_jfe2024" / "replication_package" / "data" / "ai_firm_map_2021.dta")
    bab["L1_AI_WORKER"] = bab.aiempl / bab.totalempl.where(bab.totalempl > 0); bab["fy"] = bab.year + 1
    m["gvkey_i"] = pd.to_numeric(m.gvkey, errors="coerce")
    m = m.merge(bab[["gvkey", "fy", "L1_AI_WORKER"]].rename(columns={"gvkey": "gvkey_i"}), on=["gvkey_i", "fy"], how="left")

    m = m.merge(pd.read_csv(core.out("litigation_firm_year.csv")), on="adsh", how="left")

    m["AI_MODE"] = m.industry.map(core.AI_MODE)
    m["MODE_PRODUCER"] = (m.AI_MODE == "producer").astype(int); m["MODE_CODEV"] = (m.AI_MODE == "co-developer").astype(int)

    m["LOG_WORDS"] = np.log(m.n_words.where(m.n_words > 0))
    m["is_operating"] = (((m.AT.fillna(0) >= core.MIN_ASSETS) | (m.REV.fillna(0) >= core.MIN_ASSETS)) & (m.n_words >= core.MIN_WORDS)).astype(int)
    m = m.sort_values(["cik", "fy"]).reset_index(drop=True)
    adj = m.groupby("cik").fy.shift(1) == m.fy - 1
    for c in ("C", "CAP", "G", "F", "AI_ANY"):
        if c in m: m["L1_" + c] = m.groupby("cik")[c].shift(1).where(adj)
    m.to_parquet(core.out("panel.parquet"), index=False); m.to_csv(core.out("panel.csv"), index=False)

    key = ["AI_ANY", "C", "G", "F", "L1_RD_SALES0", "L1_RD_MISSING", "L1_RD_STOCK_AT_w", "L1_RD_MKTCAP_w", "L1_LOG_PAT_STOCK",
           "L1_LOG_AI_PAT_STOCK", "L1_AI_WORKER", "PRIOR_SUIT", "IND_LIT_RATE", "REV_GROWTH_1_w", "REV_GROWTH_2_w", "REV_GROWTH_3_w",
           "D_OP_MARGIN_1_w", "L1_SIZE", "L1_LEV_w", "L1_CASH_AT_w", "L1_CAPEX_AT_w", "L1_ROA_w"]
    key = [k for k in key if k in m]
    op = m[m.is_operating == 1]
    cov = pd.DataFrame({"all": m[key].notna().mean().round(3), "operating": op[key].notna().mean().round(3)})
    cov.to_csv(core.out("panel_coverage.csv"))
    print(f"panel: {len(m):,} 10-Ks, {m.cik.nunique():,} firms | operating {len(op):,} | coded {m.coded.mean():.1%}")
    print(cov.to_string())

if __name__ == "__main__":
    main()
