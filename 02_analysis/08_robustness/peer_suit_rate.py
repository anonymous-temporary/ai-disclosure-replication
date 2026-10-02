import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import zipfile, warnings
import numpy as np
import pandas as pd
import core
import panel_est as c2
warnings.filterwarnings("ignore")

def main():
    z = zipfile.ZipFile(core.RAW / "firm_ai_measures" / "hoberg_phillips" / "tnic3_data.zip")
    T = pd.read_csv(z.open("tnic3_data.txt"), sep="\t"); T = T[T.year >= 2012]
    xw = pd.read_csv(core.out("gvkey_cik_crosswalk.csv"))[["gvkey", "cik"]].drop_duplicates("gvkey")
    suits = pd.read_csv(core.out("suits_firm_level.csv")); suits["year"] = pd.to_datetime(suits.date).dt.year
    sued = suits.merge(xw, on="cik")[["gvkey", "year"]].drop_duplicates(); sued["sued"] = 1
    P = T.merge(sued.rename(columns={"gvkey": "gvkey2"}), on=["gvkey2", "year"], how="left").fillna({"sued": 0})
    R = P.groupby(["gvkey1", "year"]).agg(PEER_LIT_RATE=("sued", "mean"), N_PEERS=("gvkey2", "size")).reset_index().rename(columns={"gvkey1": "gvkey"})
    last = R.year.max(); print(f"TNIC-3 pairs {len(T):,}, firm-years with a peer rate {len(R):,}, last TNIC year {last}", flush=True)
    R["fy"] = R.year + 1
    ext = [R]
    for y in range(last + 2, 2026):
        e = R[R.year == last].copy(); e["fy"] = y; ext.append(e)
    R = pd.concat(ext).merge(xw, on="gvkey")[["cik", "fy", "PEER_LIT_RATE", "N_PEERS"]]
    R.to_csv(core.out("peer_lit_rate.csv"), index=False)

    m = pd.read_parquet(core.out("panel.parquet")).merge(R, on=["cik", "fy"], how="left")
    d = m[(m.is_operating == 1) & m.fy.between(2015, 2025) & (m.coded == 1)].copy()
    d["ind_year"] = d.industry + "_" + d.fy.astype(int).astype(str)
    for c in ("C", "G", "F"): d["ln_" + c] = np.log1p(d[c])
    c2.TXT.append(f"peer suit rate available for {d.PEER_LIT_RATE.notna().mean():.1%} of firm-years; mean {d.PEER_LIT_RATE.mean():.3f} (industry rate mean {d.IND_LIT_RATE.mean():.3f}); corr {d[['PEER_LIT_RATE','IND_LIT_RATE']].corr().iloc[0,1]:.2f}\n")
    for fe in ("WITHIN", "BETWEEN"):
        for res in ("L1_RD_SALES0", "L1_LOG_AI_PAT_STOCK"):
            s = (d[d.fy <= 2024] if "PAT" in res else d).copy()
            for L in ("PEER_LIT_RATE", "IND_LIT_RATE"):
                dd = s.copy(); dd["RxL"] = dd[res] * dd[L]
                c2.est(dd, "ln_C", list(dict.fromkeys(c2.RD + [res, L, "RxL"] + c2.CTRL)), fe, f"H3 {res} x {L}", show=[res, L, "RxL"])
            dd = s.copy(); dd["RxP"] = dd[res] * dd.PEER_LIT_RATE; dd["RxI"] = dd[res] * dd.IND_LIT_RATE
            c2.est(dd, "ln_C", list(dict.fromkeys(c2.RD + [res, "PEER_LIT_RATE", "IND_LIT_RATE", "RxP", "RxI"] + c2.CTRL)), fe, f"H3 {res} x peer AND industry rate", show=[res, "PEER_LIT_RATE", "IND_LIT_RATE", "RxP", "RxI"])
        for y in ("ln_C", "ln_G", "ln_F"):
            c2.est(d, y, ["PEER_LIT_RATE", "IND_LIT_RATE", "PRIOR_SUIT"] + c2.RD + c2.CTRL, fe, "suit rates -> language", show=["PEER_LIT_RATE", "IND_LIT_RATE", "PRIOR_SUIT"])
    pd.DataFrame(c2.ROWS).to_csv(core.out("peer_suit_rate.csv"), index=False)
    (core.out("peer_suit_rate.txt")).write_text("".join(c2.TXT), encoding="utf-8"); print("".join(c2.TXT))

if __name__ == "__main__":
    main()
