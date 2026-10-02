import re, gzip, time, warnings
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from config import META, TEXT, sec_session, sec_get, RATE_SLEEP

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{nod}"
SKIP_DOC = re.compile(r"(^ex[-_ ]?\d|[-_]ex[-_ ]?\d|^R\d+\.htm|FilingSummary|index|"
                      r"cert|graphic|logo)", re.I)
UUENCODED = re.compile(r"^begin \d{3} .*?^end$", re.M | re.S)

def pick_document(session, cik, nod, adsh):
    items = sec_get(session, f"{ARCHIVE.format(cik=cik, nod=nod)}/index.json").json()["directory"]["item"]
    htm = [i for i in items
           if i["name"].lower().endswith((".htm", ".html")) and not SKIP_DOC.search(i["name"])]
    if htm:
        htm.sort(key=lambda i: int(i["size"] or 0), reverse=True)
        return htm[0]["name"], "htm"
    txt = [i for i in items
           if i["name"].lower().endswith(".txt") and "index" not in i["name"].lower()]
    if not txt:
        raise RuntimeError("no candidate primary document")
    txt.sort(key=lambda i: (i["name"] != f"{adsh}.txt", -int(i["size"] or 0)))
    return txt[0]["name"], "txt"

def primary_doc(session, cik, nod):
    return pick_document(session, cik, nod, None)[0]

def html_to_text(raw):
    soup = BeautifulSoup(raw, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    return re.sub(r"[ \t\xa0]+", " ", soup.get_text(" ")).strip()

def sgml_to_text(raw):
    txt = raw.decode("utf-8", "ignore")
    m = re.search(r"<DOCUMENT>\s*<TYPE>10-K[^<]*(.*?)</DOCUMENT>", txt, re.S | re.I)
    if m:
        txt = m.group(1)
    txt = UUENCODED.sub(" ", txt)
    return html_to_text(txt.encode("utf-8"))

def main():
    uni = pd.read_csv(META / "industries_10k_universe.csv")
    s, log = sec_session(), []
    total = len(uni)

    for n, (_, r) in enumerate(uni.iterrows(), 1):
        cik, adsh, fy = int(r.cik), r.adsh, int(r.fy)
        dest = TEXT / f"{cik}_{fy}_{adsh}.txt.gz"

        if dest.exists():
            log.append({"cik": cik, "fy": fy, "adsh": adsh, "status": "cached",
                        "file": dest.name, "n_words": None})
            continue

        nod = adsh.replace("-", "")
        try:
            doc, kind = pick_document(s, cik, nod, adsh)
            raw = sec_get(s, f"{ARCHIVE.format(cik=cik, nod=nod)}/{doc}", timeout=180).content
            txt = html_to_text(raw) if kind == "htm" else sgml_to_text(raw)
            if len(txt.split()) < 3000:
                raise RuntimeError(f"suspiciously short document ({len(txt.split())} words)")
            with gzip.open(dest, "wt", encoding="utf-8") as fh:
                fh.write(txt)
            log.append({"cik": cik, "fy": fy, "adsh": adsh, "status": "ok",
                        "file": dest.name, "n_words": len(txt.split())})
            if n % 25 == 0 or n == total:
                print(f"  [{n:>4}/{total}] {r['name'][:34]:<34} FY{fy}  ok", flush=True)
        except Exception as e:
            log.append({"cik": cik, "fy": fy, "adsh": adsh, "status": f"FAIL: {type(e).__name__}",
                        "file": None, "n_words": None})
            print(f"  [{n:>4}/{total}] {r['name'][:34]:<34} FY{fy}  FAILED -- {e}", flush=True)
        time.sleep(RATE_SLEEP)

    lg = pd.DataFrame(log)
    lg.to_csv(META / "download_log.csv", index=False)
    print("\n" + lg.status.str.split(":").str[0].value_counts().to_string())
    print(f"Panel now holds {len(list(TEXT.glob('*.txt.gz'))):,} filings.")

if __name__ == "__main__":
    main()
