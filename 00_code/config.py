import io, os, time, requests
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
RAW       = ROOT / "01_raw_data"
TENK      = RAW / "sec_10k"
META      = TENK / "sec_metadata"
TEXT      = TENK / "filings_text"
EXT       = RAW
for p in (RAW, TENK, META, TEXT):
    p.mkdir(parents=True, exist_ok=True)

USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
RATE_SLEEP = 0.25

def sec_session():
    if not USER_AGENT:
        raise SystemExit("Set the environment variable SEC_USER_AGENT to a name and a contact e-mail address (SEC fair-access policy).")
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return s

def sec_get(session, url, timeout=90, retries=4, **kw):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout, **kw)
            if r.status_code in (200, 206):
                return r
            if r.status_code in (403, 429, 502, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"SEC request failed after {retries} attempts: {url}")

class HTTPRangeFile(io.RawIOBase):
    def __init__(self, session, url):
        self.s, self.url, self.pos = session, url, 0
        self.size = int(session.head(url, timeout=60, allow_redirects=True)
                        .headers["Content-Length"])
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def tell(self):     return self.pos
    def seekable(self): return True
    def readable(self): return True
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0: return b""
        end = min(self.pos + n, self.size) - 1
        r = sec_get(self.s, self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        self.pos = end + 1
        return r.content

INDUSTRIES = {
    "Construction":           list(range(1500, 1800)) + [8711],
    "Construction machinery": [3523, 3531, 3532, 3537],
    "Auto manufacturing":     [3711, 3713, 3714, 3715, 3716, 3751],
    "Software & IT services": list(range(7370, 7380)),
    "Computers & chips":      list(range(3570, 3580)) + [3661, 3663, 3669, 3672, 3674, 3677, 3678, 3679],
    "Pharma & biotech":       [2834, 2835, 2836, 8731],
    "Utilities":              [4900, 4911, 4923, 4924, 4931, 4932, 4941, 4991],
    "Retail":                 list(range(5200, 6000)),
    "Aerospace & defense":    [3720, 3721, 3724, 3728, 3730, 3760, 3812],
}
SIC_TO_INDUSTRY = {sic: ind for ind, sics in INDUSTRIES.items() for sic in sics}

FY_MIN, FY_MAX = 2014, 2025
FORMS          = ("10-K", "10-K/A", "10-KT")

def industry_of(sic):
    return SIC_TO_INDUSTRY.get(int(sic)) if sic == sic else None
