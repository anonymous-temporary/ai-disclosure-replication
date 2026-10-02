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
    for c in ("C", "CAP", "G", "F"): d["ln_" + c] = np.log1p(d[c])
    d = d.sort_values(["cik", "fy"])
    d["FIRST_C"] = ((d.C > 0) & (d.groupby("cik").C.cummax().shift(1).fillna(0) == 0)).astype(int)
    c2.TXT.append(f"d5: {len(d):,} operating firm-years, FY2015-2025, three-coder labels\n")

    c2.TXT.append("\n===== A. H2 congruence: resource x industry mode -> specific claims C =====\n")
    for res in ("L1_RD_SALES0", "L1_LOG_PAT_STOCK", "L1_LOG_AI_PAT_STOCK"):
        dd = (d[d.fy <= 2024] if "PAT" in res else d).copy()
        for mode in ("INTERNAL_DEV", "HIGH_AIIE"):
            dd["RxM"] = dd[res] * dd[mode]
            xs = list(dict.fromkeys(c2.RD + [res, mode, "RxM"] + c2.CTRL))
            for fe in ("BETWEEN", "WITHIN"):
                c2.est(dd, "ln_C", xs, fe, f"A: {res} x {mode}", show=[res, mode, "RxM"])
    c2.TXT.append("\n===== B. industry suit rate and the language mix (C, G, F) =====\n")
    for y in ("ln_C", "ln_G", "ln_F"):
        c2.est(d, y, ["IND_LIT_RATE", "PRIOR_SUIT"] + c2.RD + c2.CTRL, "WITHIN", "B: suit rate -> language", show=["IND_LIT_RATE", "PRIOR_SUIT"])
    for res in ("L1_RD_SALES0", "L1_LOG_AI_PAT_STOCK"):
        dd = (d[d.fy <= 2024] if "PAT" in res else d).copy(); dd["RxL"] = dd[res] * dd.IND_LIT_RATE
        for y in ("ln_G", "ln_F"):
            c2.est(dd, y, list(dict.fromkeys(c2.RD + [res, "IND_LIT_RATE", "RxL"] + c2.CTRL)), "WITHIN", f"B: {res} x suit rate -> {y}", show=[res, "IND_LIT_RATE", "RxL"])
    c2.TXT.append("\n===== C. validation: specific claims at t -> AI patents granted at t+2, t+3 =====\n")
    pat = pd.read_csv(core.out("patents_firm_year.csv"))[["cik", "year", "N_AI_PAT", "N_PAT"]]
    for h in (2, 3):
        p = pat.rename(columns={"year": "fy", "N_AI_PAT": f"AIPAT_{h}", "N_PAT": f"PAT_{h}"}).copy(); p["fy"] = p.fy - h
        d = d.merge(p, on=["cik", "fy"], how="left")
        d[f"ln_AIPAT_{h}"] = np.log1p(d[f"AIPAT_{h}"]); d[f"any_AIPAT_{h}"] = (d[f"AIPAT_{h}"] > 0).astype(float)
        d.loc[d[f"AIPAT_{h}"].isna(), f"any_AIPAT_{h}"] = np.nan
    base = c2.RD + ["L1_LOG_AI_PAT_STOCK", "L1_LOG_PAT_STOCK"] + c2.CTRL
    for h in (2, 3):
        dd = d[d.fy <= 2023 - h].copy(); dd["NOPAT"] = (dd.L1_AI_PAT_STOCK == 0).astype(int); dd["CxNOPAT"] = dd.ln_C * dd.NOPAT
        for y in (f"ln_AIPAT_{h}", f"any_AIPAT_{h}"):
            for fe in ("BETWEEN", "WITHIN"):
                c2.est(dd, y, ["ln_C", "ln_CAP", "ln_G"] + base, fe, f"C: claims -> AI patents t+{h}", show=["ln_C", "ln_CAP", "ln_G", "L1_LOG_AI_PAT_STOCK"])
            c2.est(dd, y, ["FIRST_C", "ln_C"] + base, "BETWEEN", f"C: first-ever specific claim -> AI patents t+{h}", show=["FIRST_C", "ln_C"])
            c2.est(dd, y, ["ln_C", "NOPAT", "CxNOPAT"] + base, "BETWEEN", f"C: claim without a patent record -> AI patents t+{h}", show=["ln_C", "NOPAT", "CxNOPAT"])
    pd.DataFrame(c2.ROWS).to_csv(core.out("supplementary_tests.csv"), index=False)
    (core.out("supplementary_tests.txt")).write_text("".join(c2.TXT), encoding="utf-8"); print("".join(c2.TXT))

if __name__ == "__main__":
    main()
