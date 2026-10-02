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
IN_HOUSE = ["Software & IT services", "Computers & chips", "Aerospace & defense", "Auto manufacturing", "Pharma & biotech"]
PURCHASING = ["Retail", "Utilities", "Construction", "Construction machinery"]
ROWS = []

def fit(d, y, x, fe):
    dd = d.dropna(subset=[y] + x + ["cik", "fy", "ind_year"]).copy(); x = [v for v in x if dd[v].nunique() > 1]
    dd["firm"] = dd.cik; dd = dd.set_index(["cik", "fy"])
    mod = (PanelOLS(dd[y], dd[x], entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False) if fe == "WITHIN"
           else PanelOLS(dd[y], dd[x], other_effects=dd[["ind_year"]].astype("category"), drop_absorbed=True, check_rank=False))
    r = mod.fit(cov_type="clustered", clusters=dd[["firm"]])
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

def main():
    m = pd.read_parquet(core.out("panel.parquet"))
    d = m[(m.is_operating == 1) & m.fy.between(2015, 2025) & (m.coded == 1)].copy()
    d["ind_year"] = d.industry + "_" + d.fy.astype(int).astype(str)
    for c in ("C", "G", "F"): d["ln_" + c] = np.log1p(d[c])
    d["PHARMA"] = (d.industry == "Pharma & biotech").astype(int)
    inds = IN_HOUSE + PURCHASING
    assert set(d.industry.unique()) == set(inds), sorted(d.industry.unique())

    t1 = []
    for ind in inds + ["All"]:
        g = d if ind == "All" else d[d.industry == ind]
        t1.append({"industry": ind, "mode": "" if ind == "All" else ("in-house" if ind in IN_HOUSE else "purchasing"), "firms": g.cik.nunique(), "firm_years": len(g),
                   "any_C": (g.C > 0).mean(), "any_G": (g.G > 0).mean(), "any_F": (g.F > 0).mean(), "mean_C": g.C.mean(),
                   "rd_rev_median": g.L1_RD_SALES0.median(), "rd_reported": 1 - g.L1_RD_MISSING.mean(),
                   "rd_positive": g.L1_RD_SALES0.dropna().gt(0).mean(), "rd_at_cap": g.L1_RD_SALES0.dropna().ge(1).mean(),
                   "ai_pat_any": g.L1_AI_PAT_STOCK.dropna().gt(0).mean(), "high_aiie": g.HIGH_AIIE.dropna().mean(), "aiie_coverage": g.HIGH_AIIE.notna().mean(),
                   "suit_rate_mean": g.IND_LIT_RATE.mean(), "prior_suit": g.PRIOR_SUIT.mean(), "aiie": g.AIIE.mean()})
    pd.DataFrame(t1).to_csv(core.out("sample_by_industry.csv"), index=False)

    DVARS = ["C", "G", "F", "L1_RD_SALES0", "L1_LOG_AI_PAT_STOCK", "L1_AI_WORKER", "HIGH_AIIE", "INTERNAL_DEV",
             "IND_LIT_RATE"] + pe.CTRL
    dd_ = d[DVARS]
    desc = pd.DataFrame({"var": DVARS, "n": dd_.notna().sum().values, "mean": dd_.mean().values, "sd": dd_.std().values,
                         "min": dd_.min().values, "max": dd_.max().values})
    corr = dd_.corr(method="pearson")
    for k, v in enumerate(DVARS, start=1): desc[f"c{k}"] = corr[v].values
    desc.to_csv(core.out("descriptives.csv"), index=False)

    for res in RES:
        s = sample_for(d, res); x = list(dict.fromkeys(pe.RD + [res] + pe.CTRL))
        for y in ("ln_C", "ln_G", "ln_F"):
            for fe in ("WITHIN", "BETWEEN"):
                r, n, f = fit(s, y, x, fe); record("T2", f"H1 {res}", fe, y, r, n, f, [res])
    for y in ("ln_C", "ln_G", "ln_F"):
        for fe in ("WITHIN", "BETWEEN"):
            r, n, f = fit(d, y, list(dict.fromkeys(pe.RD + ["L1_AI_WORKER"] + pe.CTRL)), fe); record("T2V", "H1 L1_AI_WORKER", fe, y, r, n, f, ["L1_AI_WORKER"])

    for res, mode in (("L1_LOG_AI_PAT_STOCK", "INTERNAL_DEV"), ("L1_RD_SALES0", "HIGH_AIIE")):
        s = sample_for(d, res); s["RxM"] = s[res] * s[mode]
        for fe in ("WITHIN", "BETWEEN"):
            r, n, f = fit(s, "ln_C", list(dict.fromkeys(pe.RD + [res, mode, "RxM"] + pe.CTRL)), fe)
            record("T3", f"H2 {res} x {mode}", fe, "ln_C", r, n, f, [res, "RxM"] + ([mode] if mode == "HIGH_AIIE" else []))
            if mode == "HIGH_AIIE" and fe == "WITHIN":
                cols = [res, "ln_C"] + [v for v in pe.RD + pe.CTRL if v != res]
                es = s.dropna(subset=cols)
                cov = es.groupby("industry").agg(firm_years=("adsh", "size"), with_score=("HIGH_AIIE", lambda v: v.notna().mean()),
                                                 high_share=("HIGH_AIIE", lambda v: v.dropna().mean() if v.notna().any() else float("nan"))).reset_index()
                cov.loc[len(cov)] = ["All", len(es), es.HIGH_AIIE.notna().mean(), es.HIGH_AIIE.dropna().mean()]
                cov.to_csv(core.out("aiie_coverage.csv"), index=False)

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
                r, n, f = fit(s, "ln_C", terms + controls_for(res), fe)
                for ind in inds:
                    g = s[s.industry == ind]; k = f"k{inds.index(ind)}"
                    for term, kind in ((f"R_{k}", "slope"), (f"RL_{k}", "x suit rate")):
                        if term not in r.params.index: continue
                        slopes.append({"resource": lab, "fe": fe, "spec": spec, "industry": ind, "mode": "in-house" if ind in IN_HOUSE else "purchasing",
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
    pd.DataFrame(MEF).to_csv(core.out("marginal_effects.csv"), index=False)
    pd.concat(SUITS).to_csv(core.out("suit_rate_distribution.csv"), index=False)
    dd = d.copy(); dd["any_C"], dd["any_G"], dd["any_F"] = (dd.n_C > 0).astype(int), (dd.n_G > 0).astype(int), (dd.n_F > 0).astype(int)
    grp = pd.concat([dd, dd.assign(industry="All sectors")]).groupby(["industry", "fy"])
    DF = grp.agg(n=("adsh", "size"), any_C=("any_C", "mean"), any_G=("any_G", "mean"), any_F=("any_F", "mean"), AI_ANY=("AI_ANY", "mean")).reset_index()
    DF["mode"] = np.where(DF.industry.isin(IN_HOUSE), "in-house", np.where(DF.industry == "All sectors", "", "purchasing"))
    DF.to_csv(core.out("disclosure_diffusion.csv"), index=False)
    pd.DataFrame(slopes).to_csv(core.out("sector_slopes.csv"), index=False)
    pd.DataFrame(walds).to_csv(core.out("wald.csv"), index=False)
    M = pd.DataFrame(ROWS); M.to_csv(core.out("models.csv"), index=False)

    W = pd.DataFrame(walds)
    print("Wald tests (equal slopes across industries)\n" + W.round(4).to_string(index=False))
    S = pd.DataFrame(slopes); print(S[S.spec == "H3 slopes"].round(4)[["resource", "fe", "industry", "kind", "coef", "t", "firms_with_resource"]].to_string(index=False))

if __name__ == "__main__":
    main()
