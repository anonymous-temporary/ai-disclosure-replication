import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import argparse
import numpy as np
import pandas as pd
import core
import codebook as cb

from llm_runner import MODELS, run_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all"); ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int); ap.add_argument("--batch-size", type=int)
    a = ap.parse_args(); core.ensure_dirs()
    P = pd.read_csv(core.out("passages.csv"))
    if a.pilot:
        rng = np.random.default_rng(20260921)
        P = pd.concat([g.iloc[rng.choice(len(g), size=min(len(g), 28), replace=False)] for _, g in P.groupby("industry")])
        print(f"PILOT: {len(P)} passages", flush=True)
    if a.limit: P = P.head(a.limit)
    for key in (list(MODELS) if a.model == "all" else [a.model]):
        dest = core.cache("02") / (f"pilot_{key}.csv" if a.pilot else f"codes_{key}.csv")
        done = set(pd.read_csv(dest, usecols=["passage_id"]).passage_id) if dest.exists() else set()
        todo = P[~P.passage_id.isin(done)]
        print(f"{key}: {len(done):,} cached, {len(todo):,} to code", flush=True)
        if len(todo): run_model(key, todo.reset_index(drop=True), dest, a.batch_size or MODELS[key]["bs"])
        c = pd.read_csv(dest, dtype=str)
        print("  parsed: " + " | ".join(f"{f} {c[f].notna().mean():.1%}" for f in cb.FIELDS), flush=True)

if __name__ == "__main__":
    main()
