import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "00_code"))
import core as lib
import x04_match_scac_universe as x04

ABBR = {
    "INDUSTRIES": "IND", "INDUSTRY": "IND", "INDS": "IND", "INTERNATIONAL": "INTL", "INTERNATL": "INTL", "SYSTEMS": "SYS", "SYSTEM": "SYS",
    "PHARMACEUTICALS": "PHARM", "PHARMACEUTICAL": "PHARM", "PHARMA": "PHARM", "PHARMACEUT": "PHARM", "TECHNOLOGIES": "TECH",
    "TECHNOLOGY": "TECH", "TECHNOL": "TECH", "LABORATORIES": "LAB", "LABORATORY": "LAB", "LABS": "LAB", "MANUFACTURING": "MFG",
    "MEDICAL": "MED", "THERAPEUTICS": "THERAP", "THERAPEUT": "THERAP", "BIOSCIENCES": "BIOSCI", "BIOSCIENCE": "BIOSCI",
    "SCIENCES": "SCI", "SCIENCE": "SCI", "SCIENTIFIC": "SCI", "COMMUNICATIONS": "COMM", "COMMUNICATION": "COMM", "COMMUN": "COMM",
    "ELECTRONICS": "ELECTR", "ELECTRONIC": "ELECTR", "ELECTRIC": "ELEC", "SERVICES": "SVC", "SERVICE": "SVC", "SVCS": "SVC",
    "SOLUTIONS": "SOLTN", "SOLUTION": "SOLTN", "ENTERPRISES": "ENT", "ENTERPRISE": "ENT", "PRODUCTS": "PROD", "PRODUCT": "PROD",
    "AMERICAN": "AMER", "AMERICA": "AMER", "NATIONAL": "NATL", "RESOURCES": "RES", "RESOURCE": "RES", "FINANCIAL": "FINL",
    "PROPERTIES": "PPTYS", "PARTNERS": "PRTNRS", "ENERGY": "ENERGY", "POWER": "PWR", "UTILITIES": "UTIL", "SEMICONDUCTOR": "SEMICON",
    "SEMICONDUCTORS": "SEMICON", "SOFTWARE": "SOFTWR", "NETWORKS": "NETWRK", "NETWORK": "NETWRK", "HEALTHCARE": "HLTHCR",
    "HEALTH": "HLTH", "MOTORS": "MTR", "MOTOR": "MTR", "STORES": "STORE", "RESTAURANTS": "RESTR", "RESTAURANT": "RESTR",
    "ENGINEERING": "ENGR", "CONSTRUCTION": "CONSTR", "AEROSPACE": "AEROSP", "DEFENSE": "DEF", "GENERAL": "GEN", "UNITED": "UTD",
    "DEVELOPMENT": "DEV", "MANAGEMENT": "MGMT", "EQUIPMENT": "EQUIP", "INSTRUMENTS": "INSTR", "DEVICES": "DEVICE", "AUTOMOTIVE": "AUTO",
    "GROUP": "GRP", "HOLDINGS": "HLDG", "HOLDING": "HLDG", "HLDGS": "HLDG", "CORPORATION": "CORP", "COMPANY": "CO", "INCORPORATED": "INC",
    "LIMITED": "LTD", "TRUST": "TR", "AND": "",
}
LEGAL = {"INC", "CORP", "CO", "LTD", "LLC", "LP", "PLC", "NV", "SA", "AG", "SE", "HLDG", "GRP", "TR", "CL", "A", "B", "NEW", "OLD", "DE", "ADR", "CP"}

def canon(name, strip_legal=False):
    toks = [ABBR.get(t, t) for t in x04.norm(name).split()]
    toks = [t for t in toks if t]
    if strip_legal:
        while len(toks) > 1 and toks[-1] in LEGAL:
            toks.pop()
    return " ".join(toks)

def official():
    w = lib.RAW / "wrds"
    comp = pd.read_parquet(w / "comp_company.parquet"); funda = pd.read_parquet(w / "comp_funda.parquet", columns=["gvkey", "fyear"])
    comp["cik"] = pd.to_numeric(comp.cik, errors="coerce"); comp = comp.dropna(subset=["cik"]); comp["cik"] = comp.cik.astype(int)
    n = funda.groupby("gvkey").size().rename("n_funda"); last = funda.groupby("gvkey").fyear.max().rename("last_fyear")
    comp = comp.merge(n, on="gvkey", how="left").merge(last, on="gvkey", how="left").fillna({"n_funda": 0, "last_fyear": 0})
    comp = comp.sort_values(["cik", "n_funda", "last_fyear"], ascending=[True, False, False]).drop_duplicates("cik")
    link = pd.read_parquet(w / "ccm_link.parquet").sort_values("linkdt").drop_duplicates("gvkey", keep="last")
    out = comp[["gvkey", "cik", "conm", "sic", "naics"]].merge(link[["gvkey", "permno"]], on="gvkey", how="left")
    out["gvkey"] = out.gvkey.astype(int); out["tier"] = 0; out["sources"] = "wrds_comp_company"; out["n"] = 1
    out = out.rename(columns={"conm": "matched_names", "permno": "permno_adj"})
    uni = lib.load_universe().drop_duplicates("cik").set_index("cik")
    out["industry"] = out.cik.map(uni.industry); out["edgar_names"] = out.cik.map(uni.name)
    out.to_csv(lib.out("gvkey_cik_crosswalk.csv"), index=False)
    print(f"OFFICIAL crosswalk from WRDS: {len(out):,} of {len(uni):,} universe firms ({len(out)/len(uni):.1%}); with CRSP permno {out.permno_adj.notna().sum():,}")
    print((out.groupby("industry").size() / uni.groupby("industry").size()).round(3).to_string())

def main():
    lib.ensure_dirs()
    official()

if __name__ == "__main__":
    main()
