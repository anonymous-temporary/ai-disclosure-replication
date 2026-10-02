import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import numpy as np
import pandas as pd
import core

DATA_END = pd.Timestamp("2026-08-31")

def main():
    core.ensure_dirs()
    uni = core.load_universe(); uni["D"] = pd.to_datetime(uni.filed.astype(str), format="%Y%m%d", errors="coerce")
    ciks = set(uni.cik)

    party = pd.read_parquet(core.WRDS / "audit_feed14_company_legal_party_feed.parquet",
                            columns=["legal_case_key", "company_fkey", "defendant", "case_start_date"])
    case = pd.read_parquet(core.WRDS / "audit_feed13_legal_case_feed.parquet",
                           columns=["legal_case_key", "is_category_type_41", "is_category_type_1", "case_start_date", "settlement_dollars", "outcome"])
    sca = case[(case.is_category_type_41.fillna(0) == 1) & (case.is_category_type_1.fillna(0) == 1)]
    aa = party[party.defendant.fillna(0) == 1].merge(sca[["legal_case_key"]], on="legal_case_key")
    aa["cik"] = pd.to_numeric(aa.company_fkey, errors="coerce"); aa = aa.dropna(subset=["cik"]); aa["cik"] = aa.cik.astype(int)
    aa["date"] = pd.to_datetime(aa.case_start_date, errors="coerce"); aa["source"] = "audit_analytics"
    aa = aa[aa.cik.isin(ciks)][["cik", "date", "source", "legal_case_key"]].dropna(subset=["date"])

    sc = pd.read_csv(core.RAW / "scac" / "scac_universe_match.csv"); sc["date"] = pd.to_datetime(sc.filing_date, errors="coerce")
    sc = sc.assign(source="scac", legal_case_key=sc.case_id)[["cik", "date", "source", "legal_case_key"]].dropna(subset=["date"])
    S = pd.concat([aa, sc], ignore_index=True)
    S["ym"] = S.date.dt.to_period("M")
    suits = S.sort_values("date").groupby(["cik", "ym"]).agg(date=("date", "min"), sources=("source", lambda x: "+".join(sorted(set(x))))).reset_index()
    suits.to_csv(core.out("suits_firm_level.csv"), index=False)
    print(f"securities class actions against universe firms: Audit Analytics {aa.legal_case_key.nunique():,} cases, SCAC {sc.legal_case_key.nunique():,} cases "
          f"-> {len(suits):,} firm-month suit events for {suits.cik.nunique():,} firms | both sources: {(suits.sources.str.contains('[+]')).mean():.1%}")

    by = {c: g.date.values for c, g in suits.groupby("cik")}
    rows = []
    for r in uni.dropna(subset=["D"]).itertuples(index=False):
        ds = by.get(r.cik, np.array([], dtype="datetime64[ns]")); D = np.datetime64(r.D)
        prior = ((ds >= D - np.timedelta64(3 * 365, "D")) & (ds < D)).sum()
        f1 = ((ds > D) & (ds <= D + np.timedelta64(365, "D"))).any(); f2 = ((ds > D) & (ds <= D + np.timedelta64(730, "D"))).any()
        rows.append((r.adsh, int(prior > 0), int(prior), int(f1), int(f2), int(r.D + pd.Timedelta(days=365) <= DATA_END), int(r.D + pd.Timedelta(days=730) <= DATA_END)))
    L = pd.DataFrame(rows, columns=["adsh", "PRIOR_SUIT", "N_PRIOR_SUITS", "FUTURE_SUIT_1Y", "FUTURE_SUIT_2Y", "FUTURE_WINDOW_OK_1Y", "FUTURE_WINDOW_OK_2Y"])

    uni["cal"] = uni.D.dt.year; uni["sic2"] = (uni.sic // 100).astype("Int64")
    suits["cal"] = suits.date.dt.year
    sued = suits.drop_duplicates(["cik", "cal"])[["cik", "cal"]].assign(sued=1)
    pres = uni[["cik", "cal", "industry", "sic2"]]
    y0 = int(uni.cal.min())
    per = pd.to_datetime(uni.period.astype("Int64").astype(str), format="%Y%m%d", errors="coerce").dt.year
    pres = pd.concat([pres, uni.loc[per == y0 - 1, ["cik", "industry", "sic2"]].assign(cal=y0 - 1)], ignore_index=True)
    fy = pres.drop_duplicates(["cik", "cal"])[["cik", "cal", "industry", "sic2"]].merge(sued, on=["cik", "cal"], how="left").fillna({"sued": 0})
    ind = fy.groupby(["industry", "cal"]).sued.mean().rename("rate").reset_index(); ind["cal"] += 1
    g2 = fy.groupby(["sic2", "cal"]).sued.agg(["sum", "count"]).reset_index(); g2["cal"] += 1
    base = uni[["adsh", "cik", "industry", "sic2", "cal"]].merge(ind.rename(columns={"rate": "IND_LIT_RATE"}), on=["industry", "cal"], how="left")
    base = base.merge(g2, on=["sic2", "cal"], how="left")
    own = fy.assign(cal=fy.cal + 1)[["cik", "cal", "sued"]].rename(columns={"sued": "own_sued"})
    base = base.merge(own, on=["cik", "cal"], how="left").fillna({"own_sued": 0})
    base["SIC2_LIT_RATE"] = ((base["sum"] - base.own_sued) / (base["count"] - 1)).where(base["count"] > 5)
    L = L.merge(base[["adsh", "IND_LIT_RATE", "SIC2_LIT_RATE"]], on="adsh", how="left")
    L.to_csv(core.out("litigation_firm_year.csv"), index=False)

    m = L.merge(uni[["adsh", "industry"]], on="adsh")
    print(f"\n10-K filings {len(L):,}: PRIOR_SUIT {L.PRIOR_SUIT.mean():.1%} | FUTURE_SUIT_1Y {L[L.FUTURE_WINDOW_OK_1Y == 1].FUTURE_SUIT_1Y.mean():.1%} "
          f"| FUTURE_SUIT_2Y {L[L.FUTURE_WINDOW_OK_2Y == 1].FUTURE_SUIT_2Y.mean():.1%}")
    print(m.groupby("industry")[["PRIOR_SUIT", "IND_LIT_RATE", "FUTURE_SUIT_1Y"]].mean().round(3).to_string())

if __name__ == "__main__":
    main()
