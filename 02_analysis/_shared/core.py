import gzip, re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
RAW = PROJECT / "01_raw_data"
TENK = RAW / "sec_10k"
META = TENK / "sec_metadata"
TEXT = TENK / "filings_text"
WRDS = RAW / "wrds"
ANALYSIS = PROJECT / "02_analysis"
FOLDERS = {"01": "01_ai_sentence_extraction", "02": "02_llm_coding", "03": "03_disclosure_measures", "04": "04_technological_resources",
           "05": "05_litigation_exposure", "06": "06_panel_assembly", "07": "07_hypothesis_tests_and_sector_moderation", "08": "08_robustness"}
FILE_HOME = {"filings.csv": "01", "passages.csv": "01",
             "coder_agreement.csv": "03", "passages_coded.csv": "03", "disclosure_firm_year.csv": "03",
             "fundamentals_firm_year.csv": "04", "gvkey_cik_crosswalk.csv": "04", "patent_firm_links.parquet": "04", "patents_firm_year.csv": "04",
             "suits_firm_level.csv": "05", "litigation_firm_year.csv": "05",
             "panel.parquet": "06", "panel.csv": "06", "panel_coverage.csv": "06",
             "models.csv": "07", "sector_slopes.csv": "07", "wald.csv": "07", "sample_by_industry.csv": "07", "descriptives.csv": "07",
             "aiie_coverage.csv": "07", "marginal_effects.csv": "07", "suit_rate_distribution.csv": "07", "disclosure_diffusion.csv": "07",
             "capability_classification.csv": "07", "resource_classification.csv": "07",
             "supplementary_tests.csv": "08", "supplementary_tests.txt": "08", "subsamples.csv": "08", "subsamples.txt": "08",
             "subsamples_summary.csv": "08", "peer_lit_rate.csv": "08", "peer_suit_rate.csv": "08", "peer_suit_rate.txt": "08",
             "rd_measures.csv": "08", "rd_measures.txt": "08", "sector_inference.csv": "08", "sector_inference.txt": "08"}

def results(k):
    p = ANALYSIS / FOLDERS[k] / "results"; p.mkdir(parents=True, exist_ok=True); return p

def out(name):
    if name not in FILE_HOME: raise KeyError(f"no owning analysis folder for results file {name!r}; add it to core.FILE_HOME")
    return results(FILE_HOME[name]) / name

def cache(k):
    p = ANALYSIS / FOLDERS[k] / "cache"; p.mkdir(parents=True, exist_ok=True); return p


MIN_ASSETS = 10e6
MIN_WORDS = 5000


def ensure_dirs():
    pass

def load_universe():
    return pd.read_csv(META / "industries_10k_universe.csv")

def read_filing(cik, fy, adsh):
    p = TEXT / f"{int(cik)}_{int(fy)}_{adsh}.txt.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


B0, B1 = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
LEXICON = {
    "artificial_intelligence": r"(?i:artificial[\s\-]+intelligence)",
    "ai_abbrev":               B0 + r"A\.?I\.?" + B1,
    "machine_learning":        r"(?i:machine[\s\-]+learn(?:ing|ed))",
    "deep_learning":           r"(?i:deep[\s\-]+learning)",
    "reinforcement_learning":  r"(?i:reinforcement[\s\-]+learning)",
    "neural_network":          r"(?i:neural[\s\-]+net(?:work)?s?)",
    "generative_ai":           r"(?i:generative[\s\-]+(?:a\.?i\.?|artificial[\s\-]+intelligence|models?)|(?<![a-z])gen[\s\-]?ai(?![a-z]))",
    "large_language_model":    r"(?i:large[\s\-]+language[\s\-]+models?|foundation[\s\-]+models?|chatgpt)|" + B0 + r"LLMs?" + B1 + "|" + B0 + r"GPT[\s\-]?\d?" + B1,
    "nlp":                     r"(?i:natural[\s\-]+language[\s\-]+(?:processing|understanding|generation))|" + B0 + "NLP" + B1,
    "computer_vision":         r"(?i:(?:computer|machine)[\s\-]+vision)",
    "recognition":             r"(?i:(?:image|speech|voice|facial|face|pattern)[\s\-]+recognition)",
    "cognitive_computing":     r"(?i:cognitive[\s\-]+computing)",
    "conversational_ai":       r"(?i:(?<![a-z])chat[\s\-]?bots?|virtual[\s\-]+(?:assistant|agent)s?|conversational[\s\-]+(?:ai|agent)s?)",
}
LEX_RX = {k: re.compile(v) for k, v in LEXICON.items()}
ANY_AI_RX = re.compile("|".join(f"(?:{v})" for v in LEXICON.values()))

def count_terms(text):
    return {k: len(rx.findall(text)) for k, rx in LEX_RX.items()}


ITEMS = [
    ("item1",  r"Item\s*1\s*[\.\:\-–—]?\s*Bus"),        ("item1a", r"Item\s*1A\s*[\.\:\-–—]?\s*Risk"),
    ("item1b", r"Item\s*1B\s*[\.\:\-–—]?\s*Unresolved"), ("item2",  r"Item\s*2\s*[\.\:\-–—]?\s*Propert"),
    ("item3",  r"Item\s*3\s*[\.\:\-–—]?\s*Legal"),      ("item5",  r"Item\s*5\s*[\.\:\-–—]?\s*Market"),
    ("item7",  r"Item\s*7\s*[\.\:\-–—]?\s*Manage"),     ("item7a", r"Item\s*7A\s*[\.\:\-–—]?\s*Quantitat"),
    ("item8",  r"Item\s*8\s*[\.\:\-–—]?\s*Financial"),  ("item9a", r"Item\s*9A\s*[\.\:\-–—]?\s*Control"),
]
XREF_CUE = re.compile(r"(see|refer(red|ring)?\s+to|described|discussed|disclosed|contained|included|set\s+forth|under|within|"
                      r"in\s+Part|pursuant\s+to|listed|identified|detailed|noted|summarized|reference|incorporated)\W{0,12}$", re.I)
OPEN_QUOTE = re.compile(r"[“\"'‘(]\s*$")
TOC_MAX_GAP, MIN_SECTION = 700, 600
SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z\"'(“])")
NARRATIVE_ITEMS = ("item1", "item1a", "item1b", "item2", "item3", "item5", "item7", "item7a")

def split_items(txt):
    matches = sorted(({"item": n, "pos": m.start()} for n, pat in ITEMS for m in re.finditer(pat, txt, re.I)), key=lambda d: d["pos"])
    if not matches:
        return {"full": txt}, {"ok": False, "n_items": 0}
    toc = {m["pos"] for i, m in enumerate(matches)
           if ((matches[i + 1]["pos"] - m["pos"]) if i + 1 < len(matches) else 10 ** 9) < TOC_MAX_GAP}
    def is_xref(pos):
        before = txt[max(0, pos - 60):pos]
        return bool(XREF_CUE.search(before) or OPEN_QUOTE.search(before))
    cand = [m for m in matches if m["pos"] not in toc and not is_xref(m["pos"])]
    chosen, cursor = {}, -1
    for name, _ in ITEMS:
        for m in cand:
            if m["item"] == name and m["pos"] > cursor + MIN_SECTION:
                chosen[name] = m["pos"]; cursor = m["pos"]; break
    if len(chosen) < 4:
        return {"full": txt}, {"ok": False, "n_items": len(chosen)}
    ordered = sorted(chosen.items(), key=lambda kv: kv[1])
    return ({name: txt[pos:(ordered[i + 1][1] if i + 1 < len(ordered) else len(txt))] for i, (name, pos) in enumerate(ordered)},
            {"ok": True, "n_items": len(ordered)})
