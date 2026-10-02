import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import os, hashlib
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import core

MIN_LEN, MAX_LEN = 40, 1200

def _worker(meta):
    txt = core.read_filing(meta["cik"], meta["fy"], meta["adsh"])
    if txt is None:
        return None, []
    terms = core.count_terms(txt)
    secs, diag = core.split_items(txt)
    rec = {"cik": int(meta["cik"]), "fy": int(meta["fy"]), "adsh": meta["adsh"], "name": meta["name"], "industry": meta["industry"],
           "sic": meta.get("sic"), "filed": meta.get("filed"), "n_words": len(txt.split()), "segmented": int(diag["ok"]),
           "n_ai_terms": sum(terms.values()), "AI_ANY": int(sum(terms.values()) > 0), **{"t_" + k: v for k, v in terms.items()}}
    if not rec["AI_ANY"]:
        return rec, []
    if not diag["ok"]:
        secs = {"unsegmented": txt}
    out = []
    for sec, body in secs.items():
        if sec not in core.NARRATIVE_ITEMS and sec != "unsegmented":
            continue
        sents = [x.strip() for x in core.SENT_SPLIT.split(body)]
        for i, x in enumerate(sents):
            if not (MIN_LEN < len(x) < MAX_LEN and core.ANY_AI_RX.search(x)):
                continue
            s = " ".join(x.split())
            out.append({"cik": rec["cik"], "fy": rec["fy"], "adsh": rec["adsh"], "industry": rec["industry"], "section": sec,
                        "sentence_index": i, "sentence": s,
                        "context_before": " ".join((sents[i - 1] if i else "").split())[:500],
                        "context_after": " ".join((sents[i + 1] if i + 1 < len(sents) else "").split())[:500],
                        "terms": "|".join(k for k, rx in core.LEX_RX.items() if rx.search(s))})
    return rec, out

def main():
    core.ensure_dirs()
    metas = [r.to_dict() for _, r in core.load_universe().iterrows()]
    print(f"reading {len(metas):,} filings ...", flush=True)
    rows, P = [], []
    with ProcessPoolExecutor(max_workers=max(4, (os.cpu_count() or 8) - 4)) as ex:
        for i, (rec, ps) in enumerate(ex.map(_worker, metas, chunksize=24), 1):
            if rec: rows.append(rec); P.extend(ps)
            if i % 4000 == 0: print(f"    {i:,} filings, {len(P):,} passages", flush=True)
    F = pd.DataFrame(rows); P = pd.DataFrame(P).drop_duplicates(["cik", "fy", "sentence"]).reset_index(drop=True)
    P.insert(0, "passage_id", ["P" + hashlib.sha1(f"{r.cik}|{r.fy}|{r.sentence}".encode()).hexdigest()[:10] for r in P.itertuples()])
    F.to_csv(core.out("filings.csv"), index=False); P.to_csv(core.out("passages.csv"), index=False)
    print(f"\nfilings {len(F):,} | AI_ANY {F.AI_ANY.mean():.1%} | passages {len(P):,} | segmented {F.segmented.mean():.1%}")
    print("AI_ANY by fiscal year:"); print(F.groupby("fy").AI_ANY.mean().round(3).to_string())
    print("term families (share of filings with the term):"); print((F[[c for c in F.columns if c.startswith('t_')]] > 0).mean().round(3).to_string())

if __name__ == "__main__":
    main()
