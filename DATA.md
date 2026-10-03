# Data

The code in this repository builds the study’s data from public and licensed sources and estimates every model reported in the paper. This file states where each input comes from, where the scripts expect it, and which derived data the repository includes. The code runs on Python 3.12; the libraries are listed in `requirements.txt`.

## Layout

| Folder | Contents |
|---|---|
| `00_code/` | acquisition scripts, which write to `01_raw_data/` |
| `01_raw_data/` | not included: create it at the repository root and fill it as listed under Sources |
| `02_analysis/` | the analysis, one folder per step, run in numeric order; every script writes to a `results/` folder next to it |
| `data/` | the study’s own derived data, gzip-compressed |

## Sources

| Location under `01_raw_data/` | Files read by the analysis | Source | Access | Obtained with |
|---|---|---|---|---|
| `sec_10k/sec_metadata/` | `all_submissions.parquet`, `industries_10k_universe.csv`, `firm_financials.csv` | SEC Financial Statement Data Sets (https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets) and the XBRL company facts API (https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | public | `s01`, `s02`, `s03b` |
| `sec_10k/filings_text/` | one gzipped plain-text file per 10-K, `<cik>_<fiscal year>_<accession>.txt.gz` | SEC EDGAR (https://www.sec.gov/Archives/edgar/data/) | public | `s03` |
| `sec_xbrl_extra/` | `firm_financials_extra.csv` | SEC XBRL company facts | public | `s03c` |
| `sec_tickers/` | `company_tickers.json` | https://www.sec.gov/files/company_tickers.json (snapshot of 17 September 2026 in the study) | public | download |
| `wrds/` | `comp_company`, `comp_funda`, `ccm_link`, `audit_feed13_legal_case_feed`, `audit_feed14_company_legal_party_feed` (parquet) | Compustat, CRSP/Compustat Merged and Audit Analytics, through WRDS (https://wrds-www.wharton.upenn.edu) | subscription | `x08` |
| `scac/` | `scac_cases.csv`, `scac_match_manual.csv`, `scac_universe_match.csv` | Stanford Securities Class Action Clearinghouse (https://securities.stanford.edu) | public case pages | `x01`, then `x04`; `scac_match_manual.csv` is in `data/` |
| `patentsview/` | `g_patent.tsv.zip`, `g_assignee_disambiguated.tsv.zip` | PatentsView granted patent tables, now on the USPTO Open Data Portal (https://data.uspto.gov/bulkdata/datasets/pvgpatdis) | public | download |
| `uspto_aipd/` | `ai_model_predictions.csv.zip` | USPTO Artificial Intelligence Patent Dataset, 2023 release (https://www.uspto.gov/ip-policy/economic-research/research-datasets/artificial-intelligence-patent-dataset) | public | download |
| `kpss/` | `KPSS_2025.zip` | Kogan, Papanikolaou, Seru and Stoffman (2017), data extended through 2025 (https://github.com/KPSS2017/Technological-Innovation-Resource-Allocation-and-Growth-Extended-Data) | public | download |
| `firm_ai_measures/discern2/` | `output_files/csv_files/permno_gvkey.csv`, `output_files/csv_files/discern_pat_grant_1980_2021.csv` | DISCERN 2 (Arora, Belenzon and Sheer, 2021), https://zenodo.org/records/13153196 | public | `x05`, then extract the zip in place |
| `firm_ai_measures/hoberg_phillips/` | `tnic3_data.zip` | Hoberg and Phillips TNIC-3 industries (https://hobergphillips.tuck.dartmouth.edu) | public | `x05` |
| `ai_exposure_sector/aioe_felten/` | `AIOE-main.zip` | Felten, Raj and Seamans (2021), AI industry exposure (https://github.com/AIOE-Data/AIOE) | public | `x05` |
| `babina_jfe2024/` | `replication_package/data/ai_firm_map_2021.dta` | Babina, Fedyk, He and Hodson (2024), replication package (https://data.mendeley.com/datasets/s26kxvspn7/3) | public | download; extract and rename the top folder of the package to `replication_package` |

The study used the versions of these files available in September 2026; the public files were downloaded on 17 September 2026, and later releases may differ slightly. The SEC scripts send the User-Agent that the SEC fair-access policy requires (https://www.sec.gov/os/accessing-edgar-data): set the environment variable `SEC_USER_AGENT` to a name and a contact e-mail address. `x08` reads the WRDS user name from `WRDS_USERNAME`. Compustat, CRSP and Audit Analytics data are licensed and cannot be redistributed, so the firm-year panel, which joins them to the disclosure measures, is not included.

## Order of execution

1. `00_code/`: `s01_fetch_sec_metadata.py`, `s02_build_universe.py`, `s03_download_filings.py`, `s03b_fetch_financials.py`, `s03c_fetch_financials_extra.py`, `x01_scrape_scac_cases.py`, `x04_match_scac_universe.py`, `x05_fetch_public_files.py`, `x08_fetch_wrds.py`.
2. `02_analysis/01_ai_sentence_extraction/a1_passages.py`: candidate AI sentences from Items 1 to 7A of every 10-K.
3. `02_analysis/02_llm_coding/a3_code_passages.py`: the three coders, Qwen2.5-7B, Ministral-8B and Phi-4, with 4-bit weights and greedy decoding on one CUDA GPU with 16 GB of memory (model weights from Hugging Face).
4. `02_analysis/03_disclosure_measures/a4_disclosure_vars.py`: majority labels, the 74 corrections in `audit_corrections.csv`, and the measures C, G and F per 10-K.
5. `02_analysis/04_technological_resources/`: `b3_gvkey_cik.py`, then `b1_fundamentals.py` and `b4_patents.py`.
6. `02_analysis/05_litigation_exposure/b2_litigation.py`.
7. `02_analysis/06_panel_assembly/c1_panel.py`.
8. `02_analysis/07_hypothesis_tests_and_sector_moderation/analyze.py`: the estimates of the main tables and figures.
9. `02_analysis/08_robustness/`: `peer_suit_rate.py`, `rd_measures.py`, `sector_inference.py` (the key estimates under firm, two-way and sector clustering, and the wild cluster bootstrap by sector), `subsamples.py` and `supplementary_tests.py`, in any order after step 8.

## Derived data in `data/`

| File | Content | Produced by | Location expected by the scripts |
|---|---|---|---|
| `filings.csv.gz` | one row per 10-K (22,780 filings of 3,778 firms): word count, AI term counts, item segmentation flag | `a1` | `02_analysis/01_ai_sentence_extraction/results/filings.csv` |
| `passages.csv.gz` | the 72,681 candidate AI sentences, each with the sentence before and after | `a1` | `02_analysis/01_ai_sentence_extraction/results/passages.csv` |
| `codes_qwen.csv.gz`, `codes_ministral.csv.gz`, `codes_phi4.csv.gz` | each coder’s reply to every sentence, raw and parsed | `a3` | `02_analysis/02_llm_coding/cache/codes_<model>.csv` |
| `passages_coded.csv.gz` | every sentence with its majority labels after the corrections; 66,928 sentences about AI, 3,021 specific capability claims | `a4` | an output, for reading |
| `disclosure_firm_year.csv.gz` | the counts and the measures C, G and F per 10-K, per 10,000 words | `a4` | an output, for reading |
| `scac_match_manual.csv` | accept or reject decisions, with the CIK, for the 609 near-miss name matches `x04` lists for review | review of the `x04` list | `01_raw_data/scac/scac_match_manual.csv` |

With the first five files decompressed to the locations shown, the pipeline can start at step 4 without the 10-K texts or a GPU; steps 5 to 7 still need the inputs listed under Sources.
