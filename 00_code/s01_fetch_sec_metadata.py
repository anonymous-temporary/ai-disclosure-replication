import zipfile, time, sys
import pandas as pd
from config import META, sec_session, HTTPRangeFile, RATE_SLEEP

BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{}.zip"
QUARTERS = [f"{y}q{q}" for y in range(2015, 2027) for q in range(1, 5)]

def main():
    s, frames = sec_session(), []
    for qtr in QUARTERS:
        try:
            z   = zipfile.ZipFile(HTTPRangeFile(s, BASE.format(qtr)))
            sub = pd.read_csv(z.open("sub.txt"), sep="\t", low_memory=False)
            sub["source_quarter"] = qtr
            frames.append(sub)
            print(f"  {qtr}: {len(sub):>6,} filings", flush=True)
        except Exception as e:
            print(f"  {qtr}: unavailable ({type(e).__name__})", flush=True)
        time.sleep(RATE_SLEEP)

    if not frames:
        sys.exit("No quarters downloaded -- check network/User-Agent.")

    allsub = pd.concat(frames, ignore_index=True)
    dest   = META / "all_submissions.parquet"
    allsub.to_parquet(dest, index=False)
    print(f"\nSaved {len(allsub):,} filing records -> {dest}")

if __name__ == "__main__":
    main()
