import re, json
from collections import defaultdict
import pandas as pd
from rapidfuzz import process, fuzz
from config import EXT, META

SCAC  = EXT / "scac"
PRIOR_YEARS, FUTURE_YEARS = 3, 1
SUIT_MIN_YEAR = 2010
FUZZY_ACCEPT, FUZZY_REVIEW = 93, 78

SUFFIX = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "COMPANIES",
          "LTD", "LIMITED", "LLC", "LP", "PLC", "NV", "SA", "AG", "SE", "AB", "ASA",
          "HOLDINGS", "HOLDING", "HLDGS", "GROUP", "GRP", "TRUST", "PARTNERS", "ENTERPRISES"}

def norm(s):
    s = str(s).upper()
    s = re.sub(r"\\[A-Z ]+\\?", " ", s)
    s = re.sub(r"/[A-Z ]{2,}/?$", " ", s)
    s = s.replace("&", " AND ").replace("+", " AND ")
    s = re.sub(r"[.']", "", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"^(THE|IN RE) ", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()

def core(s):
    toks = norm(s).split()
    while len(toks) > 1 and toks[-1] in SUFFIX:
        toks.pop()
    return " ".join(toks)

def scac_company(case_name):
    return scac_names(case_name)[0]

def scac_names(case_name):
    s = re.sub(r"\s*(Securities\s+)+Litigation\s*$", "", case_name, flags=re.I)
    s = re.sub(r"\s*Litigation\s+Securities\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)
    s = s.split(" : ")[0].strip()
    alts = []
    m = re.match(r"^(.*?)\s*\((?:f/k/a|formerly|now|n/k/a|a/k/a|fka|nka)\s+(.*?)\)\s*(.*)$", s, flags=re.I)
    if m:
        alts = [(m.group(1) + " " + m.group(3)).strip(), m.group(2).strip()]
    else:
        m = re.match(r"^(.*?),\s*(?:formerly|now|f/k/a|n/k/a)\s+(.*)$", s, flags=re.I)
        alts = [m.group(1).strip(), m.group(2).strip()] if m else [s]
    alts = [re.sub(r"\s*\([A-Z0-9.&' -]{1,12}\)\s*$", "", a).strip() for a in alts]
    return [a for a in alts if a] or [s]

def build_dictionary():
    uni = pd.read_csv(META / "industries_10k_universe.csv")
    ciks = set(uni.cik.astype(int))
    names = defaultdict(set)
    for r in uni[["cik", "name"]].drop_duplicates().itertuples(index=False):
        names[int(r.cik)].add(r.name)
    sub = pd.read_parquet(META / "all_submissions.parquet", columns=["cik", "name", "former", "changed", "filed"])
    sub = sub[sub.cik.isin(ciks)]
    win = defaultdict(lambda: [pd.Timestamp("2100-01-01"), pd.Timestamp("1900-01-01")])
    def use(n, cik, d):
        w = win[(n, cik)]; w[0], w[1] = min(w[0], d), max(w[1], d)
    for r in uni[["cik", "name", "filed"]].drop_duplicates().itertuples(index=False):
        use(r.name, int(r.cik), pd.to_datetime(str(int(r.filed)), format="%Y%m%d"))
    for r in sub.drop_duplicates().itertuples(index=False):
        names[int(r.cik)].add(r.name)
        use(r.name, int(r.cik), pd.to_datetime(str(int(r.filed)), format="%Y%m%d"))
        if isinstance(r.former, str) and r.former.strip():
            names[int(r.cik)].add(r.former)
            end = pd.to_datetime(str(int(r.changed)), format="%Y%m%d", errors="coerce") if pd.notna(r.changed) else pd.NaT
            use(r.former, int(r.cik), pd.Timestamp("1990-01-01"))
            use(r.former, int(r.cik), end if pd.notna(end) else pd.Timestamp("2013-12-31"))
    tk = json.load(open(EXT / "sec_tickers" / "company_tickers.json", encoding="utf-8"))
    for v in (tk.values() if isinstance(tk, dict) else tk):
        if int(v["cik_str"]) in ciks:
            names[int(v["cik_str"])].add(v["title"])
    ind = uni.drop_duplicates("cik").set_index("cik").industry.to_dict()
    formers = set(sub.former.dropna().astype(str))
    tenk = set(uni.name) | set(sub.name)
    cik_win = {}
    for (n, cik), w in win.items():
        if n in tenk:
            cw = cik_win.setdefault(cik, [w[0], w[1]]); cw[0], cw[1] = min(cw[0], w[0]), max(cw[1], w[1])
    full, stem, squash = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for cik, ns in names.items():
        for n in sorted(ns, key=lambda n: (n, cik) not in win):
            pr = 2 if (n in formers and n not in tenk) else 1
            own = win.get((n, cik))
            w = own or cik_win.get(cik) or [pd.Timestamp("1990-01-01"), pd.Timestamp("2026-12-31")]
            for key, d in ((norm(n), full), (core(n), stem), (core(n).replace(" ", ""), squash)):
                if key:
                    old = d[key].get(cik)
                    if old and not own:
                        d[key][cik] = (min(pr, old[0]), old[1], old[2])
                    else:
                        d[key][cik] = (min(pr, old[0]), min(w[0], old[1]), max(w[1], old[2])) if old else (pr, w[0], w[1])
    return names, full, stem, squash, ind

NUMTOK = re.compile(r"^(\d+|I{1,3}|IV|V|VI{0,3}|IX|X)$")

def pick(cands, when=None):
    best = min(v[0] for v in cands.values())
    top = [c for c, v in cands.items() if v[0] == best]
    if len(top) == 1:
        return top[0]
    if when is None or pd.isna(when):
        return None
    def dist(c):
        lo, hi = cands[c][1] - pd.DateOffset(years=1), cands[c][2] + pd.DateOffset(years=1)
        return 0 if lo <= when <= hi else min(abs((when - lo).days), abs((when - hi).days))
    ranked = sorted(top, key=dist)
    return ranked[0] if dist(ranked[0]) < dist(ranked[1]) and dist(ranked[0]) <= 730 else None

def main():
    names, full, stem, squash, ind = build_dictionary()
    sc = pd.read_csv(SCAC / "scac_cases.csv")
    sc["company"] = sc.case_name.map(scac_company)
    sc["filing_dt"] = pd.to_datetime(sc.filing_date, errors="coerce")
    n_all = len(sc)
    sc = sc[sc.filing_dt.dt.year >= SUIT_MIN_YEAR]
    stem_keys = list(stem)
    out, review = [], []
    for r in sc.itertuples(index=False):
        tier = cik = score = None
        for cand_name in scac_names(r.case_name):
            f, c = norm(cand_name), core(cand_name)
            if f in full and pick(full[f], r.filing_dt) is not None:
                tier, cik, score = 1, pick(full[f], r.filing_dt), 100; break
            if c in stem and pick(stem[c], r.filing_dt) is not None:
                tier, cik, score = 2, pick(stem[c], r.filing_dt), 100; break
            q = c.replace(" ", "")
            if q in squash and pick(squash[q], r.filing_dt) is not None:
                tier, cik, score = 2, pick(squash[q], r.filing_dt), 100; break
            if len(c) >= 6:
                hits = process.extract(c, stem_keys, scorer=fuzz.ratio, limit=3, score_cutoff=FUZZY_REVIEW)
                if not hits:
                    continue
                best, sc1, _ = hits[0]
                runner = hits[1][1] if len(hits) > 1 else 0
                ct, bt = c.split(), best.split()
                same_num = {t for t in ct if NUMTOK.match(t)} == {t for t in bt if NUMTOK.match(t)}
                first_ok = ct[0] == bt[0] or (min(len(ct[0]), len(bt[0])) >= 4 and (ct[0].startswith(bt[0]) or bt[0].startswith(ct[0])))
                ok = (sc1 >= FUZZY_ACCEPT and first_ok and abs(len(ct) - len(bt)) <= 1 and same_num
                      and pick(stem[best], r.filing_dt) is not None and sc1 > runner)
                if ok:
                    tier, cik, score = 3, pick(stem[best], r.filing_dt), sc1; break
                review.append({"case_id": r.case_id, "scac_company": cand_name, "filing_date": r.filing_date,
                               "candidate": best, "candidate_ciks": ";".join(map(str, stem[best].keys())), "score": round(sc1, 1),
                               "runner_up": hits[1][0] if len(hits) > 1 else "", "runner_score": round(runner, 1)})
        if cik is not None:
            out.append({"case_id": r.case_id, "case_name": r.case_name, "scac_company": r.company,
                        "filing_date": r.filing_date, "status": r.status, "cik": cik,
                        "edgar_names": " | ".join(sorted(names[cik])), "industry": ind.get(cik, ""),
                        "tier": tier, "score": round(score, 1)})
    manual = (pd.read_csv(SCAC / "scac_match_manual.csv") if (SCAC / "scac_match_manual.csv").exists()
              else pd.DataFrame(columns=["case_id", "decision", "cik"]))
    auto_ids = {o["case_id"] for o in out}
    for r in sc[sc.case_id.isin(manual[manual.decision.eq("accept")].case_id)].itertuples(index=False):
        if r.case_id in auto_ids:
            continue
        cik = int(manual.set_index("case_id").cik[r.case_id])
        out.append({"case_id": r.case_id, "case_name": r.case_name, "scac_company": r.company,
                    "filing_date": r.filing_date, "status": r.status, "cik": cik,
                    "edgar_names": " | ".join(sorted(names[cik])), "industry": ind.get(cik, ""),
                    "tier": 4, "score": 100})
    reviewed = set(manual.case_id)
    review = [x for x in review if x["case_id"] not in reviewed]
    m = pd.DataFrame(out); m.to_csv(SCAC / "scac_universe_match.csv", index=False)
    (pd.DataFrame(review).sort_values("score", ascending=False) if review else pd.DataFrame(columns=["case_id"])
     ).to_csv(SCAC / "scac_match_review.csv", index=False)
    print(f"manual verdicts applied: {manual.decision.value_counts().to_dict()}; unreviewed near-misses left: {len(review)}")
    print(f"{len(m):,} of {len(sc):,} cases filed {SUIT_MIN_YEAR}+ (of {n_all:,} total) matched to {m.cik.nunique()} firms   "
          f"tiers: {m.tier.value_counts().sort_index().to_dict()}   review list: {len(review)}")
    m["year"] = pd.to_datetime(m.filing_date, errors="coerce").dt.year
    print("\nmatched cases filed 2011+ by industry:\n" + m[m.year >= 2011].groupby("industry").size().to_string())
    print("\nmatched cases by year (2011+):\n" + m[m.year >= 2011].year.value_counts().sort_index().to_string())

    uni = pd.read_csv(META / "industries_10k_universe.csv")
    uni["filed_dt"] = pd.to_datetime(uni.filed.astype(str), format="%Y%m%d", errors="coerce")
    suits = m.assign(dt=pd.to_datetime(m.filing_date, errors="coerce")).dropna(subset=["dt"]).groupby("cik").dt.apply(list).to_dict()
    def flags(r):
        ds = suits.get(int(r.cik), [])
        prior = any(r.filed_dt - pd.DateOffset(years=PRIOR_YEARS) <= d < r.filed_dt for d in ds)
        future = any(r.filed_dt < d <= r.filed_dt + pd.DateOffset(years=FUTURE_YEARS) for d in ds)
        return pd.Series({"PRIOR_SUIT": int(prior), "FUTURE_SUIT": int(future)})
    fy = uni[["adsh", "cik", "fy", "industry", "filed_dt"]].dropna(subset=["filed_dt"])
    fy = pd.concat([fy, fy.apply(flags, axis=1)], axis=1)
    fy.to_csv(SCAC / "scac_firm_year_preview.csv", index=False)
    print(f"\nfirm-years {len(fy):,}: PRIOR_SUIT {fy.PRIOR_SUIT.mean():.1%}  FUTURE_SUIT {fy.FUTURE_SUIT.mean():.1%}")
    print(fy.groupby("industry")[["PRIOR_SUIT", "FUTURE_SUIT"]].mean().round(3).to_string())

if __name__ == "__main__":
    main()
