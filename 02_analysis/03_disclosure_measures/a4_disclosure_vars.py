import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import sys
from itertools import combinations
import pandas as pd
import core

from labels import SPEC_MIN, KEYS, vote

def main():
    files = sorted(core.cache("02").glob("codes_*.csv"))
    if not files: sys.exit("no ballots yet")
    B = pd.concat([pd.read_csv(f, dtype=str) for f in files]).drop_duplicates(["passage_id", "model"], keep="last")
    models = sorted(B.model.unique())
    P = pd.read_csv(core.out("passages.csv"))
    nb = B.groupby("passage_id").model.nunique()
    print(f"ballots from {models}; passages with >=1 ballot {len(nb):,} of {len(P):,}; with all {len(models)}: {(nb == len(models)).sum():,}")

    out = pd.DataFrame(index=nb.index)
    for fld in ("about", "cap", "tone", "risk"):
        out[fld] = vote(B.pivot(index="passage_id", columns="model", values=fld))
    det = B.dropna(subset=["detail"]).copy()
    det = det[det.detail.str.len() == 6]
    for i, k in enumerate(KEYS): det[k] = det.detail.str[i].astype(int)
    dm = det.groupby("passage_id")[KEYS].mean()
    for k in KEYS: out["d_" + k.lower()] = (dm[k] > 0.5).astype(int).reindex(out.index)
    AC = pd.read_csv(_Path(__file__).with_name("audit_corrections.csv")).set_index("passage_id")
    assert AC.index.isin(out.index).all(), "an audit correction names a passage without ballots"
    not_ai = AC.index[AC.verdict.isin(["NOT_AI", "TEXT"])]; not_cap = AC.index[AC.verdict.isin(["NOT_OWN", "NOT_CLAIM"])]
    out.loc[not_ai, "about"] = "NOT_AI"
    out.loc[AC.index, "cap"] = "NONE"
    out.loc[AC.index, ["d_" + k.lower() for k in KEYS]] = 0
    print(f"audit corrections applied: {len(not_ai)} passages set to not about AI, {len(not_cap)} to no capability")
    out["is_ai"] = (out.about == "AI").astype(int)
    out["is_cap"] = (out.cap.isin(["USE", "INTENT"]) & (out.is_ai == 1)).astype(int)
    out["spec_score"] = out[["d_" + k.lower() for k in KEYS]].sum(axis=1).where(out.is_cap == 1)
    out["is_C"] = ((out.is_cap == 1) & (out.spec_score >= SPEC_MIN)).astype(int)
    out["is_G"] = ((out.risk == "GENERIC") & (out.is_ai == 1)).astype(int)
    out["is_F"] = ((out.risk == "SPECIFIC") & (out.is_ai == 1)).astype(int)
    out["favorable"] = ((out.tone == "FAVORABLE") & (out.is_cap == 1)).astype(int)

    agr = []
    for fld in ("about", "cap", "tone", "risk"):
        pv = B.pivot(index="passage_id", columns="model", values=fld)
        for a, b in combinations(models, 2):
            both = pv[[a, b]].dropna(); agr.append({"field": fld, "pair": f"{a}-{b}", "n": len(both), "agreement": round((both[a] == both[b]).mean(), 4)})
        if len(models) >= 3:
            full = pv.dropna(); agr.append({"field": fld, "pair": "unanimous", "n": len(full), "agreement": round((full.nunique(axis=1) == 1).mean(), 4)})
            agr.append({"field": fld, "pair": "no_majority", "n": len(out), "agreement": round((out[fld] == "NO_MAJORITY").mean(), 4)})
    A = pd.DataFrame(agr); A.to_csv(core.out("coder_agreement.csv"), index=False)
    if len(A): print(A.to_string(index=False))

    L = P.merge(out.reset_index(), on="passage_id", how="inner")
    L.to_csv(core.out("passages_coded.csv"), index=False)
    ai = L[L.is_ai == 1]
    print(f"\nsentences about AI {len(ai):,} ({1 - len(ai)/len(L):.1%} dropped as NOT_AI or no majority)")
    print("capability:", ai.cap.value_counts(normalize=True).round(3).to_dict(), "| risk:", ai.risk.value_counts(normalize=True).round(3).to_dict())
    print("specificity score of capability sentences:", ai[ai.is_cap == 1].spec_score.value_counts(normalize=True).sort_index().round(3).to_dict())
    print("both a capability and a risk in one sentence:", round(((ai.is_cap == 1) & ((ai.is_G == 1) | (ai.is_F == 1))).mean(), 3))

    F = pd.read_csv(core.out("filings.csv"), usecols=["cik", "fy", "adsh", "n_words"])
    g = ai.groupby("adsh")
    fy = pd.DataFrame({"n_ai": g.size(), "n_cap": g.is_cap.sum(), "n_use": g.cap.apply(lambda s: (s == "USE").sum()),
                       "n_intent": g.cap.apply(lambda s: (s == "INTENT").sum()), "n_C": g.is_C.sum(), "pts": g.spec_score.sum(),
                       "CAP_SPEC_MEAN": g.spec_score.mean(), "n_G": g.is_G.sum(), "n_F": g.is_F.sum()}).reset_index()
    coded_adsh = set(L.adsh)
    D = F.merge(fy, on="adsh", how="left")
    cnt = [c for c in fy.columns if c.startswith("n_") or c == "pts"]
    D[cnt] = D[cnt].fillna(0)
    per = 10_000 / D.n_words.where(D.n_words > 0)
    for src, dst in (("n_C", "C"), ("pts", "C_PTS"), ("n_cap", "CAP"), ("n_use", "CAP_USE"), ("n_intent", "CAP_INTENT"), ("n_G", "G"), ("n_F", "F")):
        D[dst] = D[src] * per
    D["AI_ANY"] = (D.n_ai > 0).astype(int)
    had_passages = set(P.adsh)
    D["coded"] = (~D.adsh.isin(had_passages) | D.adsh.isin(coded_adsh)).astype(int)
    D.to_csv(core.out("disclosure_firm_year.csv"), index=False)
    m = D[(D.AI_ANY == 1)]
    print(f"\nfirm-years {len(D):,} | coded so far {D.coded.mean():.1%} | AI_ANY {D.AI_ANY.mean():.1%} | among AI firm-years: any capability {(m.n_cap > 0).mean():.1%}, "
          f"any SPECIFIC capability claim C {(m.n_C > 0).mean():.1%}, any generic risk G {(m.n_G > 0).mean():.1%}, any firm-specific risk F {(m.n_F > 0).mean():.1%}")

if __name__ == "__main__":
    main()
