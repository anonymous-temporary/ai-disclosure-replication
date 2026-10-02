import os, sys
from pathlib import Path
import pandas as pd
import wrds
from config import EXT, META

OUT = EXT / "wrds"; OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent

def username():
    f = HERE / ".wrds_username"
    u = f.read_text(encoding="utf-8").strip() if f.exists() else os.environ.get("WRDS_USERNAME", "")
    if not u:
        sys.exit("Put your WRDS username in 00_code/.wrds_username (one line) or set WRDS_USERNAME.")
    return u

def save(df, name):
    p = OUT / f"{name}.parquet"
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    df.to_parquet(p, index=False); print(f"  {name}: {len(df):,} rows -> {p.name}", flush=True)

def have(name):
    return (OUT / f"{name}.parquet").exists()

def main():
    db = wrds.Connection(wrds_username=username())
    uni = pd.read_csv(META / "industries_10k_universe.csv")
    ciks = sorted(set(uni.cik.astype(int)))
    cik10 = [f"{c:010d}" for c in ciks]

    inv = []
    for lib in ("comp", "crsp", "audit", "comp_execucomp", "wrdsapps"):
        try:
            for t in db.list_tables(library=lib): inv.append({"library": lib, "table": t})
        except Exception as e:
            inv.append({"library": lib, "table": f"NOT ACCESSIBLE: {type(e).__name__}"})
    pd.DataFrame(inv).to_csv(OUT / "wrds_tables.csv", index=False); print(f"  inventory: {len(inv)} tables", flush=True)

    if not have("comp_company"):
        q = "select gvkey, cik, conm, conml, sic, naics, gsector, ggroup, state, fic, ipodate, dldte, dlrsn from comp.company where cik in %(c)s"
        save(db.raw_sql(q, params={"c": tuple(cik10)}), "comp_company")
    comp = pd.read_parquet(OUT / "comp_company.parquet"); gv = sorted(set(comp.gvkey.dropna()))
    print(f"  universe CIKs {len(ciks):,} -> gvkeys {len(gv):,}", flush=True)

    if not have("ccm_link"):
        q = ("select gvkey, lpermno as permno, lpermco as permco, linktype, linkprim, linkdt, linkenddt "
             "from crsp.ccmxpf_lnkhist where gvkey in %(g)s and linktype in ('LU','LC','LS')")
        save(db.raw_sql(q, params={"g": tuple(gv)}, date_cols=["linkdt", "linkenddt"]), "ccm_link")
    link = pd.read_parquet(OUT / "ccm_link.parquet"); permnos = sorted(set(link.permno.dropna().astype(int)))
    print(f"  permnos {len(permnos):,}", flush=True)

    if not have("comp_funda"):
        cols = ("gvkey, datadate, fyear, fyr, cik, conm, sich, at, sale, revt, xrd, xsga, cogs, oiadp, oibdp, ib, ni, emp, capx, che, dltt, dlc, "
                "lt, ceq, seq, csho, prcc_f, mkvalt, ppent, intan, gdwl, act, lct, xad, dvc, aqc")
        q = (f"select {cols} from comp.funda where gvkey in %(g)s and fyear >= 2011 "
             "and indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'")
        save(db.raw_sql(q, params={"g": tuple(gv)}, date_cols=["datadate"]), "comp_funda")

    try:
        legal = [t for t in db.list_tables(library="audit") if any(k in t.lower() for k in ("legal", "litig", "lawsuit", "case"))]
    except Exception as e:
        legal = []; print(f"  audit library not accessible: {type(e).__name__}", flush=True)
    for t in legal:
        name = f"audit_{t}"
        if have(name): continue
        try:
            cols = list(db.describe_table(library="audit", table=t).name)
            ckey = next((c for c in ("company_fkey", "cik", "company_cik", "defendant_cik") if c in cols), None)
            if ckey:
                df = db.raw_sql(f"select * from audit.{t} where {ckey} in %(c)s", params={"c": tuple(cik10)})
                if df.empty:
                    df = db.raw_sql(f"select * from audit.{t} where {ckey} in %(c)s", params={"c": tuple(str(c) for c in ciks)})
            else:
                n = int(db.raw_sql(f"select count(*) as n from audit.{t}").n.iloc[0])
                if n > 3_000_000: print(f"  {name}: skipped, {n:,} rows and no CIK column", flush=True); continue
                df = db.raw_sql(f"select * from audit.{t}")
            save(df, name)
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
    db.close(); print("done")

if __name__ == "__main__":
    main()
