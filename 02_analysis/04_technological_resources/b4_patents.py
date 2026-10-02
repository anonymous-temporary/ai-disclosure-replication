import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import sys, zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "00_code"))
import core as lib
import b3_gvkey_cik as p3
import x04_match_scac_universe as x04

RAW = lib.RAW
Y0, Y1 = 2000, 2023

def zcsv(path, **kw):
    z = zipfile.ZipFile(path); return pd.read_csv(z.open(z.namelist()[0]), **kw)

def main():
    lib.ensure_dirs()
    xw = pd.read_csv(lib.out("gvkey_cik_crosswalk.csv"))
    g2c = dict(zip(xw.gvkey, xw.cik))

    print("patents ...", flush=True)
    pat = zcsv(RAW / "patentsview" / "g_patent.tsv.zip", sep="\t", usecols=["patent_id", "patent_type", "patent_date"], dtype=str)
    pat = pat[pat.patent_type == "utility"]
    pat["grant_year"] = pat.patent_date.str[:4].astype(int)
    pat = pat[pat.grant_year.between(Y0, Y1)][["patent_id", "grant_year"]]
    print(f"  utility patents {Y0}-{Y1}: {len(pat):,}", flush=True)

    print("AIPD ...", flush=True)
    ai = zcsv(RAW / "uspto_aipd" / "ai_model_predictions.csv.zip",
              usecols=["doc_id", "flag_patent", "predict86_any_ai", "predict93_any_ai"], dtype={"doc_id": str})
    ai = ai[ai.flag_patent == 1].rename(columns={"doc_id": "patent_id", "predict93_any_ai": "ai93", "predict86_any_ai": "ai86"})
    pat = pat.merge(ai[["patent_id", "ai93", "ai86"]], on="patent_id", how="left").fillna({"ai93": 0, "ai86": 0})

    print("KPSS values + permno ...", flush=True)
    kz = zipfile.ZipFile(RAW / "kpss" / "KPSS_2025.zip")
    kname = [n for n in kz.namelist() if n.lower().endswith(".csv")][0]
    kp = pd.read_csv(kz.open(kname)); kp.columns = [c.lower() for c in kp.columns]
    kp["patent_id"] = kp["patent_num"].astype(str)
    keep = ["patent_id"] + [c for c in ("xi_real", "xi_nominal", "cites", "permno") if c in kp.columns]
    pat = pat.merge(kp[keep].drop_duplicates("patent_id"), on="patent_id", how="left")
    if "permno" not in pat.columns:
        mp = zcsv(RAW / "kpss" / "Match_patent_permco_permno_2025.zip"); mp["patent_id"] = mp.patent_num.astype(str)
        pat = pat.merge(mp[["patent_id", "permno"]].drop_duplicates("patent_id"), on="patent_id", how="left")

    dz = RAW / "firm_ai_measures" / "discern2" / "output_files" / "csv_files"
    pg = pd.read_csv(dz / "permno_gvkey.csv")
    pg_any = pg.sort_values("max_y_permno").drop_duplicates("permno_adj", keep="last").set_index("permno_adj").gvkey
    ccm = RAW / "wrds" / "ccm_link.parquet"
    if ccm.exists():
        lk = pd.read_parquet(ccm).dropna(subset=["permno"]).sort_values("linkdt").drop_duplicates("permno", keep="last")
        pg_any = pd.concat([pg_any, pd.Series(lk.gvkey.astype(int).values, index=lk.permno.astype(int).values)])
        pg_any = pg_any[~pg_any.index.duplicated(keep="last")]
    pat["gvkey_A"] = pat.permno.map(pg_any)
    pat["cik_A"] = pat.gvkey_A.map(g2c)
    dg = pd.read_csv(dz / "discern_pat_grant_1980_2021.csv", usecols=["patent_id", "permno_adj"], dtype={"patent_id": str})
    dg = dg.dropna(subset=["permno_adj"]).drop_duplicates("patent_id")
    dg["cik_B"] = dg.permno_adj.map(pg_any).map(g2c)
    pat = pat.merge(dg[["patent_id", "cik_B"]], on="patent_id", how="left")

    print("route C: assignee names ...", flush=True)
    names, *_ = x04.build_dictionary()
    full, stem = defaultdict(set), defaultdict(set)
    for cik, ns in names.items():
        for n in ns:
            full[p3.canon(n)].add(cik); stem[p3.canon(n, True)].add(cik)
    asg = zcsv(RAW / "patentsview" / "g_assignee_disambiguated.tsv.zip", sep="\t",
               usecols=["patent_id", "assignee_sequence", "disambig_assignee_organization"], dtype=str)
    asg = asg.dropna(subset=["disambig_assignee_organization"])
    asg = asg[asg.patent_id.isin(set(pat.patent_id))]
    orgs = asg.disambig_assignee_organization.drop_duplicates()
    def to_cik(o):
        k1 = p3.canon(o)
        if k1 in full and len(full[k1]) == 1: return next(iter(full[k1]))
        k2 = p3.canon(o, True)
        if len(k2) >= 5 and k2 in stem and len(stem[k2]) == 1: return next(iter(stem[k2]))
        return np.nan
    omap = {o: to_cik(o) for o in orgs}
    asg["cik_C"] = asg.disambig_assignee_organization.map(omap)
    c = asg.dropna(subset=["cik_C"]).sort_values("assignee_sequence").drop_duplicates("patent_id")[["patent_id", "cik_C"]]
    pat = pat.merge(c, on="patent_id", how="left")

    pat["cik"] = pat.cik_A.fillna(pat.cik_B).fillna(pat.cik_C)
    pat["route"] = np.where(pat.cik_A.notna(), "A_kpss", np.where(pat.cik_B.notna(), "B_discern", np.where(pat.cik_C.notna(), "C_name", "")))
    L = pat.dropna(subset=["cik"]).copy(); L["cik"] = L.cik.astype(int)
    agree = L[L.cik_A.notna() & L.cik_C.notna()]
    print(f"  linked patents {len(L):,} to {L.cik.nunique():,} firms | routes {L.route.value_counts().to_dict()}")
    print(f"  audit: KPSS route and name route agree on {(agree.cik_A == agree.cik_C).mean():.1%} of {len(agree):,} doubly linked patents")
    L[["patent_id", "cik", "route", "grant_year", "ai93", "ai86"] + [c for c in ("xi_real", "cites") if c in L.columns]].to_parquet(
        lib.out("patent_firm_links.parquet"), index=False)

    xi = "xi_real" if "xi_real" in L.columns else None
    L["ai_xi"] = L[xi].where(L.ai93 == 1) if xi else np.nan
    fy = L.groupby(["cik", "grant_year"]).agg(N_PAT=("patent_id", "size"), N_AI_PAT=("ai93", "sum"), N_AI_PAT86=("ai86", "sum"),
                                              PAT_VALUE=(xi, "sum") if xi else ("ai93", "size"), AI_PAT_VALUE=("ai_xi", "sum")).reset_index()
    fy = fy.rename(columns={"grant_year": "year"})
    grid = pd.MultiIndex.from_product([fy.cik.unique(), range(Y0, Y1 + 1)], names=["cik", "year"]).to_frame(index=False)
    fy = grid.merge(fy, on=["cik", "year"], how="left").fillna(0).sort_values(["cik", "year"])
    for flow, stock in (("N_PAT", "PAT_STOCK"), ("N_AI_PAT", "AI_PAT_STOCK"), ("N_AI_PAT86", "AI_PAT_STOCK86"), ("AI_PAT_VALUE", "AI_PAT_VALUE_STOCK")):
        out = []
        for _, g in fy.groupby("cik", sort=False):
            k = 0.0
            for v in g[flow]:
                k = k * 0.85 + v; out.append(k)
        fy[stock] = out
    fy["AI_PAT_SHARE"] = (fy.AI_PAT_STOCK / fy.PAT_STOCK).where(fy.PAT_STOCK > 0)
    fy["LOG_AI_PAT_STOCK"] = np.log1p(fy.AI_PAT_STOCK)
    fy.to_csv(lib.out("patents_firm_year.csv"), index=False)

    uni = lib.load_universe().drop_duplicates("cik")[["cik", "industry"]]
    last = fy[fy.year == Y1].merge(uni, on="cik")
    print(f"\nfirms with any linked patent: {fy.cik.nunique():,} | with AI patent stock > 0 in {Y1}: {(last.AI_PAT_STOCK > 0).sum():,}")
    print(last.groupby("industry").agg(firms=("cik", "size"), with_ai=("AI_PAT_STOCK", lambda s: (s > 0).sum()),
                                       ai_share=("AI_PAT_SHARE", "median"), ai_stock_mean=("AI_PAT_STOCK", "mean")).round(3).to_string())
    print("\nAI patents granted per year (universe firms, 93 % threshold):")
    print(fy.groupby("year").N_AI_PAT.sum().astype(int).loc[2012:].to_string())

if __name__ == "__main__":
    main()
