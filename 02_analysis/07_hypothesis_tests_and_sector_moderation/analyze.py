import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.panel import PanelOLS
import core
import panel_est as pe
warnings.filterwarnings("ignore")

RES = {"L1_RD_SALES0": "R&D / revenue", "L1_LOG_AI_PAT_STOCK": "log AI patent stock"}
MODE = core.AI_MODE
MKEY = {"producer": "P", "co-developer": "C", "adopter": "A"}
ALT_MODES = {"as in the paper": MODE,
             "pharma as adopter": dict(MODE, **{"Pharma & biotech": "adopter"}),
             "construction machinery as adopter": dict(MODE, **{"Construction machinery": "adopter"}),
             "two groups (build, buy)": {k: ("adopter" if k in ("Retail", "Utilities", "Construction", "Construction machinery") else "producer") for k in MODE}}
CLASSES = ["Specific capability claim", "Capability statement below three criteria", "AI risk only", "Other AI mention", "No AI language"]
CRITERIA = [("d_action", "Action"), ("d_usecase", "Use case"), ("d_named", "Named product or unit"), ("d_quant", "Quantity"),
            ("d_timing", "Date or stage"), ("d_verifiable", "Verifiable detail"), ("is_C", "Three or more criteria")]
RD_TIERS = ["None or not reported", "Under 5% of revenue", "5% of revenue or more", "Pre-revenue"]
WF_TIERS = ["No AI workers", "Under .5% of employees", ".5% of employees or more"]
TECH = [("hardware", "AI hardware"), ("planning", "Planning and control"), ("kr", "Knowledge processing"), ("nlp", "Natural language processing"),
        ("vision", "Vision"), ("ml", "Machine learning"), ("speech", "Speech"), ("evo", "Evolutionary computation")]
ROWS = []

def fit(d, y, x, fe, two_way=True):
    dd = d.dropna(subset=[y] + x + ["cik", "fy", "ind_year"]).copy(); x = [v for v in x if dd[v].nunique() > 1]
    dd["firm"] = dd.cik; dd["sector_year"] = pd.factorize(dd.ind_year)[0]; dd = dd.set_index(["cik", "fy"])
    mod = (PanelOLS(dd[y], dd[x], entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=True) if fe == "WITHIN"
           else PanelOLS(dd[y], dd[x], other_effects=dd[["ind_year"]].astype("category"), drop_absorbed=True, check_rank=True))
    r = mod.fit(cov_type="clustered", clusters=dd[["firm", "sector_year"]] if two_way else dd[["firm"]])
    return r, len(dd), dd["firm"].nunique()

def record(table, model, fe, y, r, n, firms, show):
    for k in show:
        if k in r.params.index:
            ROWS.append({"table": table, "model": model, "fe": fe, "y": y, "term": k, "coef": r.params[k], "se": r.std_errors[k],
                         "t": r.tstats[k], "p": r.pvalues[k], "n": n, "firms": firms, "r2_within": r.rsquared})

def controls_for(res):
    return [v for v in pe.RD if v != res] + pe.CTRL if res == "L1_RD_SALES0" else pe.RD + pe.CTRL

def sample_for(d, res):
    return (d[d.fy <= 2024] if "PAT" in res else d).copy()

def mode_fits(s, res, assign, label):
    ms = [k for k in core.MODES if k in set(assign.values())]; s = s.copy(); g = s.industry.map(assign); out = []
    for k in ms: s[f"M_{MKEY[k]}"] = (g == k).astype(float); s[f"Rx{MKEY[k]}"] = s[res] * s[f"M_{MKEY[k]}"]
    for fe in ("WITHIN", "BETWEEN"):
        for ref in ms:
            oth = [k for k in ms if k != ref]
            x = list(dict.fromkeys([res] + [f"M_{MKEY[k]}" for k in oth] + [f"Rx{MKEY[k]}" for k in oth] + controls_for(res)))
            r, n, f = fit(s, "ln_C", x, fe)
            holders = int(s.loc[(g == ref) & (s[res] > 0), "cik"].nunique())
            out.append({"resource": RES_ALL[res], "assignment": label, "fe": fe, "kind": "slope", "label": ref, "coef": r.params[res], "se": r.std_errors[res],
                        "t": r.tstats[res], "p": r.pvalues[res], "n": n, "firms": f, "firms_with_resource": holders, "r2_within": r.rsquared})
            for k in oth:
                if ms.index(k) < ms.index(ref):
                    t = f"Rx{MKEY[k]}"
                    out.append({"resource": RES_ALL[res], "assignment": label, "fe": fe, "kind": "contrast", "label": f"{k} minus {ref}", "coef": r.params[t],
                                "se": r.std_errors[t], "t": r.tstats[t], "p": r.pvalues[t], "n": n, "firms": f, "firms_with_resource": np.nan, "r2_within": r.rsquared})
    return out

RES_ALL = {"L1_RD_SALES0": "R&D / revenue", "L1_LOG_AI_PAT_STOCK": "log AI patent stock", "L1_AI_WORKER": "AI-worker share"}

def main():
    m = pd.read_parquet(core.out("panel.parquet"))
    d = m[(m.is_operating == 1) & m.fy.between(2015, 2025) & (m.coded == 1)].copy()
    d["ind_year"] = d.industry + "_" + d.fy.astype(int).astype(str)
    for c in ("C", "G", "F"): d["ln_" + c] = np.log1p(d[c])
    d["PHARMA"] = (d.industry == "Pharma & biotech").astype(int)
    inds = core.SECTORS
    assert set(d.industry.unique()) == set(inds), sorted(d.industry.unique())

    t1 = []
    for ind in inds + ["All"]:
        g = d if ind == "All" else d[d.industry == ind]
        t1.append({"industry": ind, "mode": "" if ind == "All" else MODE[ind], "firms": g.cik.nunique(), "firm_years": len(g),
                   "any_C": (g.C > 0).mean(), "any_G": (g.G > 0).mean(), "any_F": (g.F > 0).mean(), "mean_C": g.C.mean(),
                   "rd_rev_median": g.L1_RD_SALES0.median(), "rd_reported": 1 - g.L1_RD_MISSING.mean(),
                   "rd_positive": g.L1_RD_SALES0.dropna().gt(0).mean(), "rd_at_cap": g.L1_RD_SALES0.dropna().ge(1).mean(),
                   "ai_pat_any": g.L1_AI_PAT_STOCK.dropna().gt(0).mean(), "ai_worker_half": g.L1_AI_WORKER.dropna().ge(.005).mean(),
                   "suit_rate_mean": g.IND_LIT_RATE.mean(), "prior_suit": g.PRIOR_SUIT.mean()})
    pd.DataFrame(t1).to_csv(core.out("sample_by_industry.csv"), index=False)

    DVARS = ["C", "G", "F", "L1_RD_SALES0", "L1_LOG_AI_PAT_STOCK", "L1_AI_WORKER", "MODE_PRODUCER", "MODE_CODEV",
             "IND_LIT_RATE"] + pe.CTRL
    dd_ = d[DVARS]
    desc = pd.DataFrame({"var": DVARS, "n": dd_.notna().sum().values, "mean": dd_.mean().values, "sd": dd_.std().values,
                         "min": dd_.min().values, "max": dd_.max().values})
    corr = dd_.corr(method="pearson")
    for k, v in enumerate(DVARS, start=1): desc[f"c{k}"] = corr[v].values
    desc.to_csv(core.out("descriptives.csv"), index=False)

    cls = pd.Series(np.select([d.n_C > 0, d.n_cap > 0, (d.n_G > 0) | (d.n_F > 0), d.AI_ANY == 1], CLASSES[:4], CLASSES[4]), index=d.index)
    S_ = pd.read_csv(core.out("passages_coded.csv"), usecols=["adsh", "is_ai", "is_cap", "is_C"] + [k for k, _ in CRITERIA[:-1]])
    S_ = S_[(S_.is_ai == 1) & (S_.is_cap == 1)].merge(d[["adsh", "industry"]], on="adsh")
    rows_ = []
    for ind in inds + ["All sectors"]:
        k10 = cls if ind == "All sectors" else cls[d.industry == ind]
        for c in CLASSES:
            rows_.append({"family": "10-K class", "category": c, "industry": ind, "count": int((k10 == c).sum()), "n": len(k10)})
        st = S_ if ind == "All sectors" else S_[S_.industry == ind]
        for col, lab in CRITERIA:
            rows_.append({"family": "capability criterion", "category": lab, "industry": ind, "count": int(st[col].sum()), "n": len(st)})
    CL = pd.DataFrame(rows_); CL["share"] = CL["count"] / CL["n"]
    assert (CL[CL.family == "10-K class"].groupby("industry")["count"].sum() == CL[CL.family == "10-K class"].groupby("industry").n.first()).all()
    assert int(CL[(CL.category == CRITERIA[-1][1]) & (CL.industry == "All sectors")]["count"].iloc[0]) == int(d.n_C.sum())
    CL.to_csv(core.out("capability_classification.csv"), index=False)

    rd_ok = d.L1_RD_SALES0.notna()
    rdc = pd.Series(np.select([(d.L1_RD_MISSING == 1) | (d.L1_RD_SALES0 == 0), d.L1_PRE_REVENUE == 1, d.L1_RD_SALES0 < .05],
                              [RD_TIERS[0], RD_TIERS[3], RD_TIERS[1]], RD_TIERS[2]), index=d.index)
    wf = d.L1_AI_WORKER
    wfc = pd.Series(np.select([wf == 0, wf < .005], WF_TIERS[:2], WF_TIERS[2]), index=d.index)
    last_sector = d.sort_values("fy").drop_duplicates("cik", keep="last").set_index("cik").industry
    PL = pd.read_parquet(core.out("patent_firm_links.parquet"), columns=["patent_id", "cik", "grant_year", "ai93"] + [f"ai93_{k}" for k, _ in TECH])
    PL = PL[(PL.ai93 == 1) & PL.grant_year.between(2014, 2023)].copy(); PL["industry"] = PL.cik.map(last_sector); PL = PL.dropna(subset=["industry"])
    sc = pd.read_csv(core.RAW / "scac" / "scac_universe_match.csv")
    sc = sc[sc.cik.isin(set(d.cik)) & pd.to_datetime(sc.filing_date, errors="coerce").dt.year.between(2013, 2025)]
    rows_ = []
    for ind in inds + ["All sectors"]:
        sel = (d.industry == ind) if ind != "All sectors" else pd.Series(True, index=d.index)
        for fam, ser, ok, cats in (("R&D intensity", rdc, rd_ok, RD_TIERS), ("AI workforce share", wfc, wf.notna(), WF_TIERS)):
            k = ser[sel & ok]
            for c in cats: rows_.append({"family": fam, "category": c, "industry": ind, "count": int((k == c).sum()), "n": len(k)})
        pp = PL if ind == "All sectors" else PL[PL.industry == ind]
        for col, lab in TECH: rows_.append({"family": "AI patent technology", "category": lab, "industry": ind, "count": int(pp[f"ai93_{col}"].sum()), "n": len(pp)})
        ps = d.loc[sel, "PRIOR_SUIT"].dropna()
        rows_.append({"family": "litigation", "category": "Sued in the prior three years", "industry": ind, "count": int(ps.sum()), "n": len(ps)})
    for st in ("DISMISSED", "SETTLED", "ONGOING"):
        rows_.append({"family": "litigation", "category": f"SCAC suit {st.lower()}", "industry": "All sectors", "count": int((sc.status == st).sum()), "n": len(sc)})
    RC = pd.DataFrame(rows_); RC["share"] = RC["count"] / RC["n"]
    for fam in ("R&D intensity", "AI workforce share"):
        f_ = RC[RC.family == fam]; assert (f_.groupby("industry")["count"].sum() == f_.groupby("industry").n.first()).all(), fam
    RC.to_csv(core.out("resource_classification.csv"), index=False)

    for res in RES:
        s = sample_for(d, res); x = list(dict.fromkeys(pe.RD + [res] + pe.CTRL))
        for y in ("ln_C", "ln_G", "ln_F"):
            for fe in ("WITHIN", "BETWEEN"):
                r, n, f = fit(s, y, x, fe); record("T2", f"H1 {res}", fe, y, r, n, f, [res])
    for y in ("ln_C", "ln_G", "ln_F"):
        for fe in ("WITHIN", "BETWEEN"):
            r, n, f = fit(d, y, list(dict.fromkeys(pe.RD + ["L1_AI_WORKER"] + pe.CTRL)), fe); record("T2V", "H1 L1_AI_WORKER", fe, y, r, n, f, ["L1_AI_WORKER"])

    MROWS = []
    for res in RES_ALL:
        by_mode = mode_fits(sample_for(d, res), res, MODE, "as in the paper"); MROWS += by_mode
        for q in by_mode:
            ROWS.append({"table": "T3", "model": f"H2 {res} x AI_MODE", "fe": q["fe"], "y": "ln_C", "term": f"{q['kind']} {q['label']}", "coef": q["coef"],
                         "se": q["se"], "t": q["t"], "p": q["p"], "n": q["n"], "firms": q["firms"], "r2_within": q["r2_within"]})
    for lab_, asg in list(ALT_MODES.items())[1:]:
        for res in RES_ALL: MROWS += mode_fits(sample_for(d, res), res, asg, lab_)
    s = sample_for(d, "L1_LOG_AI_PAT_STOCK"); s["TREND"] = s.fy - 2015; res = "L1_LOG_AI_PAT_STOCK"
    for k in core.MODES:
        s[f"M_{MKEY[k]}"] = (s.AI_MODE == k).astype(float); s[f"R_{MKEY[k]}"] = s[res] * s[f"M_{MKEY[k]}"]; s[f"RT_{MKEY[k]}"] = s[f"R_{MKEY[k]}"] * s.TREND
    for fe in ("WITHIN", "BETWEEN"):
        x = [f"R_{MKEY[k]}" for k in core.MODES] + [f"RT_{MKEY[k]}" for k in core.MODES] + ["M_P", "M_C"] + controls_for(res)
        r, n, f = fit(s, "ln_C", list(dict.fromkeys(x)), fe); last = int(s.TREND.max())
        for k in core.MODES:
            a_, b_ = f"R_{MKEY[k]}", f"RT_{MKEY[k]}"; e = r.params[a_] + last * r.params[b_]
            se = np.sqrt(r.cov.loc[a_, a_] + last ** 2 * r.cov.loc[b_, b_] + 2 * last * r.cov.loc[a_, b_])
            for kind, c_, se_, p_ in (("slope in FY2015", r.params[a_], r.std_errors[a_], r.pvalues[a_]), ("change per year", r.params[b_], r.std_errors[b_], r.pvalues[b_]),
                                      (f"slope in FY{2015 + last}", e, se, 2 * stats.t.sf(abs(e / se), r.df_resid))):
                MROWS.append({"resource": RES_ALL[res], "assignment": "as in the paper", "fe": fe, "kind": kind, "label": k, "coef": c_, "se": se_, "t": c_ / se_, "p": p_,
                              "n": n, "firms": f, "firms_with_resource": np.nan, "r2_within": r.rsquared})
    pd.DataFrame(MROWS).to_csv(core.out("mode_models.csv"), index=False)
    wv = []
    for v in ("ln_C", "L1_LOG_AI_PAT_STOCK", "L1_RD_SALES0", "L1_AI_WORKER"):
        z = d[["cik", v]].dropna(); wv.append({"variable": v, "within_share": (z[v] - z.groupby("cik")[v].transform("mean")).var() / z[v].var(), "firms": z.cik.nunique(), "n": len(z)})
    pd.DataFrame(wv).to_csv(core.out("within_variance.csv"), index=False)


    MEF, SUITS = [], []
    for res in RES:
        s = sample_for(d, res); s["RxL"] = s[res] * s.IND_LIT_RATE
        s["RxP"] = s[res] * s.PHARMA; s["LxP"] = s.IND_LIT_RATE * s.PHARMA; s["RxLxP"] = s.RxL * s.PHARMA
        for fe in ("WITHIN", "BETWEEN"):
            base = list(dict.fromkeys(pe.RD + [res, "IND_LIT_RATE", "RxL"] + pe.CTRL))
            r, n, f = fit(s, "ln_C", base, fe); record("T4", f"H3 {res} baseline", fe, "ln_C", r, n, f, [res, "IND_LIT_RATE", "RxL"])
            b_, V = r.params, r.cov; L_ = s.dropna(subset=base + ["ln_C"]).IND_LIT_RATE
            for q in np.linspace(.01, .99, 50):
                Lq = L_.quantile(q); me = b_[res] + b_["RxL"] * Lq
                se = np.sqrt(V.loc[res, res] + Lq ** 2 * V.loc["RxL", "RxL"] + 2 * Lq * V.loc[res, "RxL"])
                MEF.append({"resource": RES[res], "fe": fe, "quantile": q, "suit_rate": Lq, "effect": me, "se": se, "lo": me - 1.96 * se, "hi": me + 1.96 * se, "n": n})
            if fe == "WITHIN": SUITS.append(pd.DataFrame({"resource": RES[res], "suit_rate": L_.values}))
            r, n, f = fit(s, "ln_C", base + ["RxP", "LxP", "RxLxP"], fe); record("T4", f"H3 {res} x pharma", fe, "ln_C", r, n, f, [res, "IND_LIT_RATE", "RxL", "RxP", "LxP", "RxLxP"])

    fyear = (m.filed // 10000).astype(int)
    rate = pd.Series(m.IND_LIT_RATE.values, index=pd.MultiIndex.from_arrays([m.industry, fyear])).dropna()
    rate = rate[~rate.index.duplicated()]
    assert m.assign(fyear=fyear).dropna(subset=["IND_LIT_RATE"]).groupby(["industry", "fyear"]).IND_LIT_RATE.nunique().max() == 1
    d["L_PREV"] = rate.reindex(pd.MultiIndex.from_arrays([d.industry, (d.filed // 10000).astype(int) - 1])).values
    for res in RES:
        s = sample_for(d, res).dropna(subset=["L_PREV"]).copy(); s["RxL"] = s[res] * s.IND_LIT_RATE; s["RxLp"] = s[res] * s.L_PREV
        for fe in ("WITHIN", "BETWEEN"):
            r, n, f = fit(s, "ln_C", list(dict.fromkeys(pe.RD + [res, "IND_LIT_RATE", "RxL"] + pe.CTRL)), fe)
            record("R1", f"H3 {res} same sample as L_PREV", fe, "ln_C", r, n, f, [res, "IND_LIT_RATE", "RxL"])
            r, n, f = fit(s, "ln_C", list(dict.fromkeys(pe.RD + [res, "L_PREV", "RxLp"] + pe.CTRL)), fe)
            record("R1", f"H3 {res} L one year earlier", fe, "ln_C", r, n, f, [res, "L_PREV", "RxLp"])

    sd_L = d.IND_LIT_RATE.std(); slopes, walds = [], []
    for res, lab in RES.items():
        s = sample_for(d, res).dropna(subset=[res, "IND_LIT_RATE", "ln_C"] + controls_for(res))
        for ind in inds:
            D = (s.industry == ind).astype(float); k = f"k{inds.index(ind)}"
            s[f"R_{k}"] = s[res] * D; s[f"L_{k}"] = s.IND_LIT_RATE * D; s[f"RL_{k}"] = s[res] * s.IND_LIT_RATE * D
        for fe in ("WITHIN", "BETWEEN"):
            for spec, terms in (("H1 slopes", [f"R_k{inds.index(i)}" for i in inds]),
                                ("H3 slopes", [f"R_k{inds.index(i)}" for i in inds] + [f"L_k{inds.index(i)}" for i in inds] + [f"RL_k{inds.index(i)}" for i in inds])):
                r, n, f = fit(s, "ln_C", terms + controls_for(res), fe, two_way=False)
                for ind in inds:
                    g = s[s.industry == ind]; k = f"k{inds.index(ind)}"
                    for term, kind in ((f"R_{k}", "slope"), (f"RL_{k}", "x suit rate")):
                        if term not in r.params.index: continue
                        slopes.append({"resource": lab, "fe": fe, "spec": spec, "industry": ind, "mode": MODE[ind],
                                       "kind": kind, "coef": r.params[term], "se": r.std_errors[term], "t": r.tstats[term], "p": r.pvalues[term],
                                       "per_sd_suit": r.params[term] * sd_L if kind == "x suit rate" else np.nan,
                                       "se_per_sd_suit": r.std_errors[term] * sd_L if kind == "x suit rate" else np.nan,
                                       "firm_years": len(g), "firms_with_resource": g.loc[g[res] > 0, "cik"].nunique(),
                                       "sd_resource": g[res].std(), "suit_rate_sd_within_industry": g.IND_LIT_RATE.std(), "n_model": n})
                for kind, pre in (("slope", "R_"), ("x suit rate", "RL_")):
                    if spec == "H1 slopes" and kind != "slope": continue
                    names = [f"{pre}k{inds.index(i)}" for i in inds if f"{pre}k{inds.index(i)}" in r.params.index and s.loc[s.industry == i, res].gt(0).groupby(s.cik).any().sum() >= 20]
                    b = r.params[names].values; V = r.cov.loc[names, names].values
                    Rm = np.hstack([np.ones((len(names) - 1, 1)), -np.eye(len(names) - 1)])
                    W = float((Rm @ b) @ np.linalg.pinv(Rm @ V @ Rm.T) @ (Rm @ b)); df = len(names) - 1
                    walds.append({"resource": lab, "fe": fe, "spec": spec, "kind": kind, "industries": len(names), "chi2": W, "df": df, "p": stats.chi2.sf(W, df)})
    res = "L1_LOG_AI_PAT_STOCK"; s = sample_for(d, res).dropna(subset=[res, "IND_LIT_RATE", "ln_C"] + controls_for(res)).copy(); tv = []
    s["POST"] = (s.fy >= 2022).astype(float); s["TREND"] = s.fy - 2015
    for ind in inds:
        D = (s.industry == ind).astype(float); k = f"k{inds.index(ind)}"
        s[f"R_{k}"] = s[res] * D; s[f"L_{k}"] = s.IND_LIT_RATE * D; s[f"RL_{k}"] = s[res] * s.IND_LIT_RATE * D
        s[f"RP_{k}"] = s[f"R_{k}"] * s.POST; s[f"RT_{k}"] = s[f"R_{k}"] * s.TREND
    K = [f"k{i}" for i in range(len(inds))]; base = [f"R_{k}" for k in K] + [f"L_{k}" for k in K] + [f"RL_{k}" for k in K]
    for spec, terms, ss in (("as in the main table", base, s), ("slope shift from fiscal year 2022", base + [f"RP_{k}" for k in K], s),
                            ("slope trend by year", base + [f"RT_{k}" for k in K], s), ("fiscal years 2015 to 2021", base, s[s.fy <= 2021])):
        for fe in ("WITHIN", "BETWEEN"):
            r, n, f = fit(ss, "ln_C", terms + controls_for(res), fe, two_way=False)
            for ind in inds:
                t = f"RL_k{inds.index(ind)}"
                if t in r.params.index:
                    tv.append({"spec": spec, "fe": fe, "industry": ind, "coef": r.params[t], "se": r.std_errors[t], "t": r.tstats[t], "p": r.pvalues[t], "n_model": n,
                               "firms_with_resource": int(ss.loc[(ss.industry == ind) & (ss[res] > 0), "cik"].nunique())})
    pd.DataFrame(tv).to_csv(core.out("sector_slopes_time.csv"), index=False)
    pd.DataFrame(MEF).to_csv(core.out("marginal_effects.csv"), index=False)
    pd.concat(SUITS).to_csv(core.out("suit_rate_distribution.csv"), index=False)
    dd = d.copy(); dd["any_C"], dd["any_G"], dd["any_F"] = (dd.n_C > 0).astype(int), (dd.n_G > 0).astype(int), (dd.n_F > 0).astype(int)
    grp = pd.concat([dd, dd.assign(industry="All sectors")]).groupby(["industry", "fy"])
    DF = grp.agg(n=("adsh", "size"), any_C=("any_C", "mean"), any_G=("any_G", "mean"), any_F=("any_F", "mean"), AI_ANY=("AI_ANY", "mean")).reset_index()
    DF["mode"] = DF.industry.map(MODE).fillna("")
    DF.to_csv(core.out("disclosure_diffusion.csv"), index=False)
    pd.DataFrame(slopes).to_csv(core.out("sector_slopes.csv"), index=False)
    pd.DataFrame(walds).to_csv(core.out("wald.csv"), index=False)
    M = pd.DataFrame(ROWS); M.to_csv(core.out("models.csv"), index=False)

    W = pd.DataFrame(walds)
    print("Wald tests (equal slopes across industries)\n" + W.round(4).to_string(index=False))
    S = pd.DataFrame(slopes); print(S[S.spec == "H3 slopes"].round(4)[["resource", "fe", "industry", "kind", "coef", "t", "firms_with_resource"]].to_string(index=False))

if __name__ == "__main__":
    main()
