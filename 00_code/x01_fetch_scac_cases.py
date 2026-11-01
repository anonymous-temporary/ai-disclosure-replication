import gzip, re, time
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "01_raw_data" / "scac"
HTML = OUT / "case_html"
HTML.mkdir(parents=True, exist_ok=True)

ID_MIN, ID_MAX = 100001, 108900
STOP_AFTER_EMPTY = 60
SLEEP = 1.0
UA = "Mozilla/5.0 (academic research)"

def clean(s):
    s = re.sub(r"<[^>]+>", " ", s)
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&#8212;", "-")
           .replace("&quot;", '"').replace("&#39;", "'"))
    return re.sub(r"\s+", " ", s).strip()

def parse(cid, h):
    if "Protected Content" in h and not re.search(r"<h4>\s*\S[^<]*Securities Litigation", h):
        return None
    m = re.search(r"<h4>\s*([^<]+?)\s*</h4>", h)
    name = clean(m.group(1)) if m else ""
    if not name or name.lower().startswith("protected"):
        return None
    st  = re.search(r"Case Status:.*?</strong>\s*(?:<span[^>]*></span>)?\s*([A-Z][A-Z /]*?)\s*(?:&nbsp;|<)", h, re.S)
    sd  = re.search(r"On or around\s*([\d/]+)\s*\(([^)]*)\)", h)
    jd  = re.search(r"Presiding Judge:.*?</strong>\s*(?:<br[^>]*>)?\s*([^<]+?)\s*<", h, re.S)
    fd  = re.search(r"Filing Date:\s*([A-Za-z]+ \d{1,2}, \d{4})", h)
    body = re.search(r'<div class="span12" style="background-color: #ffffff;">(.*?)</div>', h, re.S)
    court = re.search(r"(?:District Court|Court):?\s*</(?:strong|b|td|dt)>\s*(?:<[^>]+>\s*)*([^<]{3,60})", h)
    return {"case_id": cid, "case_name": name,
            "status": clean(st.group(1)) if st else "",
            "status_date": sd.group(1) if sd else "",
            "status_note": clean(sd.group(2)) if sd else "",
            "judge": clean(jd.group(1)) if jd else "",
            "filing_date": fd.group(1) if fd else "",
            "court": clean(court.group(1)) if court else "",
            "summary": clean(body.group(1))[:3000] if body else ""}

def main():
    s = requests.Session(); s.headers["User-Agent"] = UA
    rows, empty_run, last_real = [], 0, None
    for cid in range(ID_MIN, ID_MAX + 1):
        f = HTML / f"{cid}.html.gz"
        h = None
        if f.exists():
            try:
                h = gzip.open(f, "rt", encoding="utf-8", errors="ignore").read()
            except (OSError, EOFError, Exception) as e:
                print(f"{cid} corrupt cache ({type(e).__name__}), refetching", flush=True)
                f.unlink(); h = None
        if h is None:
            try:
                r = s.get(f"https://securities.stanford.edu/filings-case.html?id={cid}", timeout=60)
                h = r.text if r.status_code == 200 else ""
            except requests.RequestException as e:
                print(f"{cid} ERROR {type(e).__name__}", flush=True); h = ""
            if h:
                with gzip.open(f, "wt", encoding="utf-8") as g: g.write(h)
            time.sleep(SLEEP)
        rec = parse(cid, h)
        if rec:
            rows.append(rec); empty_run = 0; last_real = cid
        else:
            empty_run += 1
            if last_real and cid > 108600 and empty_run >= STOP_AFTER_EMPTY:
                print(f"stop: {STOP_AFTER_EMPTY} empty ids after {last_real}"); break
        if cid % 250 == 0:
            print(f"  {cid}  cases so far: {len(rows)}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "scac_cases.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scac_cases.csv", index=False)
    print(f"\n{len(df):,} cases  ids {df.case_id.min()}..{df.case_id.max()}  "
          f"filing dates {df.filing_date.head(1).values} .. {df.filing_date.tail(1).values}")

if __name__ == "__main__":
    main()
