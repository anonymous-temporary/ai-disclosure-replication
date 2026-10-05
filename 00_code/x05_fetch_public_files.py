import re, time, csv
import requests
from config import EXT

UA = {"User-Agent": "Mozilla/5.0 (academic research)"}
LOG = EXT / "_download_log_x05.csv"

FILES = [
    ("firm_ai_measures/discern2", "DISCERN_2_0_0.zip",
     "https://zenodo.org/api/records/13153196/files/DISCERN%202_0_0.zip/content", "http"),
    ("firm_ai_measures/hoberg_phillips", "tnic3_data.zip", "https://hobergphillips.tuck.dartmouth.edu/idata/tnic3_data.zip", "http"),
]

def stream(r, dest):
    n = 0
    with open(dest, "wb") as f:
        for c in r.iter_content(1 << 20):
            f.write(c); n += len(c)
    return n

def fetch_http(url, dest):
    r = requests.get(url, headers=UA, stream=True, timeout=300, allow_redirects=True)
    r.raise_for_status()
    return stream(r, dest), r.headers.get("Content-Type", ""), r.headers.get("Content-Disposition", "")

def fetch_gdrive(url, dest):
    fid = re.search(r"/d/([^/]+)", url).group(1)
    s = requests.Session(); s.headers.update(UA)
    r = s.get(f"https://drive.google.com/uc?export=download&id={fid}", stream=True, timeout=300)
    if "text/html" in r.headers.get("Content-Type", ""):
        html = r.text
        m = re.search(r'action="([^"]+)"', html)
        params = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', html))
        if m and params:
            r = s.get(m.group(1).replace("&amp;", "&"), params=params, stream=True, timeout=300)
    r.raise_for_status()
    cd = r.headers.get("Content-Disposition", "")
    name = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd)
    real = dest.with_name(name.group(1)) if name else dest.with_suffix("")
    return stream(r, real), r.headers.get("Content-Type", ""), real.name

def main():
    done = set()
    if LOG.exists():
        done = {row["path"] for row in csv.DictReader(open(LOG, encoding="utf-8")) if row["status"] == "ok"}
    rows = []
    for sub, name, url, kind in FILES:
        d = EXT / sub; d.mkdir(parents=True, exist_ok=True)
        dest = d / name; rel = f"{sub}/{name}"
        if rel in done or (dest.exists() and dest.stat().st_size > 0) or (kind == "gdrive" and any(d.glob(name.replace(".gdrive", "*")))):
            continue
        t0 = time.time()
        try:
            if kind == "gdrive":
                n, ctype, real = fetch_gdrive(url, dest); note = f"saved as {real}"
            else:
                n, ctype, cd = fetch_http(url, dest); note = cd
            status = "ok"
            print(f"  ok   {rel:<70} {n/1e6:8.1f} MB  {time.time()-t0:5.0f}s  {ctype[:30]}", flush=True)
        except Exception as e:
            status, n, note = f"FAIL: {type(e).__name__}: {str(e)[:100]}", 0, ""
            if dest.exists(): dest.unlink()
            print(f"  FAIL {rel:<70} {status}", flush=True)
        rows.append({"path": rel, "url": url, "status": status, "bytes": n, "note": note,
                     "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "url", "status", "bytes", "note", "fetched_utc"])
        if new: w.writeheader()
        w.writerows(rows)
    print(f"\n{sum(r['status']=='ok' for r in rows)} fetched, {sum(r['status']!='ok' for r in rows)} failed, "
          f"{sum(r['bytes'] for r in rows)/1e9:.2f} GB")

if __name__ == "__main__":
    main()
