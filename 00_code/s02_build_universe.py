import pandas as pd
from config import META, SIC_TO_INDUSTRY, FY_MIN, FY_MAX, FORMS

def main():
    s = pd.read_parquet(META / "all_submissions.parquet")
    s["sic"] = pd.to_numeric(s["sic"], errors="coerce")
    s["fy"]  = pd.to_numeric(s["fy"],  errors="coerce")

    m = s[s.sic.isin(SIC_TO_INDUSTRY)
          & s.form.isin(FORMS)
          & s.fy.between(FY_MIN, FY_MAX)].copy()

    m["form_rank"] = m.form.map({"10-K": 0, "10-KT": 1, "10-K/A": 2}).fillna(3)
    m = (m.sort_values(["cik", "fy", "form_rank"])
           .drop_duplicates(["cik", "fy"], keep="first"))

    m["industry"] = m.sic.map(SIC_TO_INDUSTRY)
    cols = ["adsh", "cik", "name", "sic", "industry", "form", "fy", "fp",
            "period", "filed", "countryba", "stprba", "cityba", "source_quarter"]
    m = m[[c for c in cols if c in m.columns]].sort_values(["industry", "name", "fy"])

    dest = META / "industries_10k_universe.csv"
    m.to_csv(dest, index=False)

    print(f"Firm-year observations : {len(m):,}")
    print(f"Unique firms           : {m.cik.nunique():,}")
    print(f"Fiscal years           : {int(m.fy.min())}-{int(m.fy.max())}")
    print("\nBy industry:")
    print(m.groupby("industry").agg(firms=("cik", "nunique"), obs=("adsh", "count")).to_string())
    print(f"\nSaved -> {dest}")

if __name__ == "__main__":
    main()
