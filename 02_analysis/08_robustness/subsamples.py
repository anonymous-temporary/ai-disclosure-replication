import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import warnings
import numpy as np
import pandas as pd
import core
import panel_est as c2
warnings.filterwarnings("ignore")

def main():
    m = pd.read_parquet(core.out("panel.parquet"))
    d = m[(m.is_operating == 1) & m.fy.between(2015, 2025) & (m.coded == 1)].copy()
    d["ind_year"] = d.industry + "_" + d.fy.astype(int).astype(str)
    for c in ("C", "G", "F"): d["ln_" + c] = np.log1p(d[c])
    d["SIZE_T"] = pd.qcut(d.L1_SIZE, 3, labels=["small", "mid", "large"])
    samples = {"all": d, "no pharma": d[d.industry != "Pharma & biotech"], "no software": d[d.industry != "Software & IT services"],
               "post-ChatGPT FY>=2023": d[d.fy >= 2023], "pre-ChatGPT FY<=2022": d[d.fy <= 2022], "large firms": d[d.SIZE_T == "large"], "small firms": d[d.SIZE_T == "small"]}
    for name, s in samples.items():
        c2.TXT.append(f"\n\n################ SAMPLE: {name}  ({len(s):,} firm-years)\n")
        for fe in ("WITHIN", "BETWEEN"):
            c2.est(s, "ln_C", c2.RD + c2.CTRL, fe, f"[{name}] H1 R&D -> C", show=["L1_RD_SALES0"])
            sp = s[s.fy <= 2024]
            c2.est(sp, "ln_C", c2.RD + ["L1_LOG_AI_PAT_STOCK"] + c2.CTRL, fe, f"[{name}] H1 AI patents -> C", show=["L1_LOG_AI_PAT_STOCK"])
            c2.est(sp, "ln_G", c2.RD + ["L1_LOG_AI_PAT_STOCK"] + c2.CTRL, fe, f"[{name}] RQ1 AI patents -> G", show=["L1_LOG_AI_PAT_STOCK"])
            dd = sp.copy(); dd["RxM"] = dd.L1_LOG_AI_PAT_STOCK * dd.INTERNAL_DEV
            c2.est(dd, "ln_C", c2.RD + ["L1_LOG_AI_PAT_STOCK", "INTERNAL_DEV", "RxM"] + c2.CTRL, fe, f"[{name}] H2 AI patents x in-house", show=["RxM"])
            dd = s.copy(); dd["RxM"] = dd.L1_RD_SALES0 * dd.HIGH_AIIE
            c2.est(dd, "ln_C", c2.RD + ["HIGH_AIIE", "RxM"] + c2.CTRL, fe, f"[{name}] H2 R&D x high AI exposure", show=["RxM"])
            dd = s.copy(); dd["RxL"] = dd.L1_RD_SALES0 * dd.IND_LIT_RATE
            c2.est(dd, "ln_C", c2.RD + ["IND_LIT_RATE", "RxL"] + c2.CTRL, fe, f"[{name}] H3a R&D x industry suit rate", show=["RxL"])
            dd = sp.copy(); dd["RxL"] = dd.L1_LOG_AI_PAT_STOCK * dd.IND_LIT_RATE
            c2.est(dd, "ln_C", c2.RD + ["L1_LOG_AI_PAT_STOCK", "IND_LIT_RATE", "RxL"] + c2.CTRL, fe, f"[{name}] H3b AI patents x industry suit rate", show=["RxL"])
    R = pd.DataFrame(c2.ROWS); R.to_csv(core.out("subsamples.csv"), index=False)
    (core.out("subsamples.txt")).write_text("".join(c2.TXT), encoding="utf-8")
    piv = R[R.term.isin(["L1_RD_SALES0", "L1_LOG_AI_PAT_STOCK", "RxM", "RxL"])].copy()
    piv["test"] = piv.model.str.extract(r"\] (.*)")[0]; piv["sample"] = piv.model.str.extract(r"\[(.*?)\]")[0]
    tab = piv.pivot_table(index=["test", "fe"], columns="sample", values="t").round(1)
    tab.to_csv(core.out("subsamples_summary.csv")); print(tab.to_string())

if __name__ == "__main__":
    main()
