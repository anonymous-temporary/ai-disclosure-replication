import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import warnings
import numpy as np
import pandas as pd
import core
import panel_est as pe
warnings.filterwarnings("ignore")

def main():
    m = pd.read_parquet(core.out("panel.parquet"))
    d = m[(m.is_operating == 1) & m.fy.between(2015, 2025) & (m.coded == 1)].copy(); d["ind_year"] = d.industry + "_" + d.fy.astype(int).astype(str)
    d["ln_C"] = np.log1p(d.C)
    d["RD_RANK"] = d.groupby("ind_year").L1_RD_SALES0.rank(pct=True)
    d["RD_Z"] = (d.L1_RD_SALES0 - d.groupby("industry").L1_RD_SALES0.transform("mean")) / d.groupby("industry").L1_RD_SALES0.transform("std")
    cands = [("L1_RD_SALES0", "R&D / revenue, capped (baseline)"), ("RD_RANK", "R&D / revenue rank within industry-year"), ("RD_Z", "R&D / revenue z-score within industry"),
             ("L1_RD_AT_w", "R&D / assets"), ("L1_RD_STOCK_AT_w", "R&D stock / assets"), ("L1_LOG_RD", "log R&D expense")]
    cands = [(v, l) for v, l in cands if v in d]
    for name, s in (("all firm-years", d), ("without pharma", d[d.industry != "Pharma & biotech"])):
        pe.TXT.append(f"\n===== {name} ({len(s):,})\n")
        for v, lab in cands:
            ctrl = [c for c in pe.CTRL]; rd = [c for c in pe.RD if c != "L1_RD_SALES0"]
            for fe in ("WITHIN", "BETWEEN"):
                pe.est(s, "ln_C", [v] + rd + ctrl, fe, f"[{name}] H1 {lab}", show=[v])
            ss = s.copy(); ss["RxL"] = ss[v] * ss.IND_LIT_RATE
            pe.est(ss, "ln_C", [v, "IND_LIT_RATE", "RxL"] + rd + ctrl, "WITHIN", f"[{name}] H3a {lab} x suit rate", show=[v, "RxL"])
    R = pd.DataFrame(pe.ROWS); R.to_csv(core.out("rd_measures.csv"), index=False)
    piv = R[R.term.isin([v for v, _ in cands] + ["RxL"])].copy(); piv["sample"] = piv.model.str.extract(r"\[(.*?)\]")[0]; piv["test"] = piv.model.str.extract(r"\] (H\d\w?) ")[0]; piv["measure"] = piv.model.str.extract(r"\] H\d\w? (.*?)( x suit rate)?$")[0]
    tab = piv[(piv.test == "H1") | ((piv.test == "H3a") & (piv.term == "RxL"))].pivot_table(index=["measure"], columns=["sample", "test", "fe"], values="t").round(1)
    pe.TXT.append("\nt-statistics: H1 = resource -> C; H3a = resource x industry suit rate -> C (within)\n" + tab.to_string() + "\n")
    core.out("rd_measures.txt").write_text("".join(pe.TXT), encoding="utf-8"); print(tab.to_string())

if __name__ == "__main__":
    main()
