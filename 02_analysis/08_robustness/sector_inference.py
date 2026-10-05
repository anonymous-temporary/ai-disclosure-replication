import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from linearmodels.panel import PanelOLS
import core
import panel_est as pe
warnings.filterwarnings("ignore")

RES = {"L1_RD_SALES0": "R&D / revenue", "L1_LOG_AI_PAT_STOCK": "log AI patent stock", "L1_AI_WORKER": "AI-worker share"}
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(.5), np.sqrt(.5), 1.0, np.sqrt(1.5)])
RADEMACHER = np.array([-1.0, 1.0])
PAT = "L1_LOG_AI_PAT_STOCK"
def mode_specs(res):
    m = f"H2 {res} x AI_MODE"
    return [("H2", "T3", m, res, ["M_P", "M_A"], [("RxP", "M_P"), ("RxA", "M_A")], "RxP", "contrast producer minus co-developer", "producers minus co-developers"),
            ("H2", "T3", m, res, ["M_P", "M_C"], [("RxP", "M_P"), ("RxC", "M_C")], "RxC", "contrast co-developer minus adopter", "co-developers minus adopters"),
            ("H2", "T3", m, res, ["M_P", "M_C"], [("RxP", "M_P"), ("RxC", "M_C")], "RxP", "contrast producer minus adopter", "producers minus adopters")]
SPECS = [("H1", "T2", "H1 L1_RD_SALES0", "L1_RD_SALES0", [], [], "L1_RD_SALES0", None, ""),
         ("H1", "T2", "H1 L1_LOG_AI_PAT_STOCK", PAT, [], [], PAT, None, ""),
         ("H1", "T2V", "H1 L1_AI_WORKER", "L1_AI_WORKER", [], [], "L1_AI_WORKER", None, ""),
         *mode_specs("L1_RD_SALES0"), *mode_specs(PAT), *mode_specs("L1_AI_WORKER"),
         ("H3", "T4", "H3 L1_RD_SALES0 baseline", "L1_RD_SALES0", ["IND_LIT_RATE"], [("RxL", "IND_LIT_RATE")], "RxL", None, ""),
         ("H3", "T4", "H3 L1_LOG_AI_PAT_STOCK baseline", PAT, ["IND_LIT_RATE"], [("RxL", "IND_LIT_RATE")], "RxL", None, "")]
TERM_LABEL = {"RxL": "resource x litigation exposure", "RxP": "resource x production mode", "RxC": "resource x production mode"}


def group_demean(codes):
    n = len(codes); G = sp.csr_matrix((np.ones(n), (np.arange(n), codes)))
    cnt = np.asarray(G.sum(0)).ravel()
    return lambda v: v - G @ ((G.T @ v) / cnt[:, None])


def fe_annihilator(dd, fe):
    if fe == "BETWEEN":
        return group_demean(pd.factorize(dd["ind_year"])[0])
    dm = group_demean(pd.factorize(dd.index.get_level_values(0))[0])
    Z = dm(pd.get_dummies(dd.index.get_level_values(1), drop_first=True).values.astype(float))
    Zp = np.linalg.pinv(Z)
    return lambda v: dm(v) - Z @ (Zp @ dm(v))


def wcr_pvalue(A, C, t_obs, weights):
    G = len(A); m = len(weights); total = m ** G; hits = 0
    pw = m ** np.arange(G); step = max(1, min(total, 300_000))
    for s in range(0, total, step):
        idx = np.arange(s, min(s + step, total))
        v = weights[(idx[:, None] // pw) % m]
        t = (v @ A) / np.sqrt(((v @ C.T) ** 2).sum(1))
        hits += int((np.abs(t) >= abs(t_obs) * (1 - 1e-12)).sum())
    return hits / total, total


def main():
    m = pd.read_parquet(core.out("panel.parquet"))
    d = m[(m.is_operating == 1) & m.fy.between(2015, 2025) & (m.coded == 1)].copy()
    d["ind_year"] = d.industry + "_" + d.fy.astype(int).astype(str)
    d["ln_C"] = np.log1p(d["C"])
    for k, v in (("M_P", "producer"), ("M_C", "co-developer"), ("M_A", "adopter")): d[k] = (d.AI_MODE == v).astype(float)
    ref = pd.read_csv(core.out("models.csv"))
    rows, notes = [], []
    for hyp, table, model, res, extra, prod, term, ref_term, contrast in SPECS:
        s = (d[d.fy <= 2024] if "PAT" in res else d).copy()
        for nm, mod_ in prod: s[nm] = s[res] * s[mod_]
        base = list(dict.fromkeys(pe.RD + [res] + extra + [nm for nm, _ in prod] + pe.CTRL))
        for fe in ("WITHIN", "BETWEEN"):
            dd = s.dropna(subset=["ln_C"] + base + ["cik", "fy", "ind_year"]).copy(); x = [v for v in base if dd[v].nunique() > 1]
            dd["firm"] = dd.cik; dd["sector"] = pd.factorize(dd.industry)[0]; dd["sector_year"] = pd.factorize(dd.ind_year)[0]
            switchers = int(dd.groupby("cik").industry.nunique().gt(1).sum())
            dd = dd.set_index(["cik", "fy"])
            mod = (PanelOLS(dd["ln_C"], dd[x], entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=True) if fe == "WITHIN"
                   else PanelOLS(dd["ln_C"], dd[x], other_effects=dd[["ind_year"]].astype("category"), drop_absorbed=True, check_rank=True))
            r_f = mod.fit(cov_type="clustered", clusters=dd[["firm"]])
            r_2 = mod.fit(cov_type="clustered", clusters=dd[["firm", "sector_year"]])
            r_s = mod.fit(cov_type="clustered", clusters=dd[["sector"]])
            g = ref[(ref.table == table) & (ref.model == model) & (ref.fe == fe) & (ref.y == "ln_C") & (ref.term == (ref_term or term))].iloc[0]
            assert abs(r_2.params[term] - g.coef) < 1e-9 and abs(r_2.std_errors[term] - g.se) < 1e-9, (model, fe, "differs from the main model estimates")

            names = list(r_f.params.index); k = names.index(term); n, p = len(dd), len(names)
            M = fe_annihilator(dd, fe)
            Xt = M(dd[names].values.astype(float)); yt = M(dd[["ln_C"]].values.astype(float))[:, 0]
            Q = np.linalg.inv(Xt.T @ Xt); b = Q @ (Xt.T @ yt)
            assert np.allclose(b, r_f.params.values, rtol=1e-6, atol=1e-10), (model, fe, "FWL replica differs from PanelOLS")
            e = yt - Xt @ b; z = Xt @ Q[k]; cl = dd["sector"].values; G = cl.max() + 1
            score = np.bincount(cl, weights=z * e, minlength=G)
            keep = [j for j in range(p) if j != k]
            br = np.linalg.lstsq(Xt[:, keep], yt, rcond=None)[0]; ur = yt - Xt[:, keep] @ br
            U = np.zeros((n, G)); U[np.arange(n), cl] = ur
            W = M(U); assert np.allclose(W.sum(1), ur, atol=1e-9)
            A = z @ U
            E = W - Xt @ (Q @ (Xt.T @ W))
            C = np.vstack([z[cl == h] @ E[cl == h] for h in range(G)])
            assert abs(A.sum() - b[k]) < 1e-9 * max(1, abs(b[k])) and np.allclose(C.sum(1), score, atol=1e-12)
            t_obs = b[k] / np.sqrt((score ** 2).sum())
            p_webb, n_webb = wcr_pvalue(A, C, t_obs, WEBB)
            p_rad, n_rad = wcr_pvalue(A, C, t_obs, RADEMACHER)
            se_s = r_s.std_errors[term]
            rows.append({"hypothesis": hyp, "resource": RES[res], "design": fe, "term": TERM_LABEL.get(term, "resource"), "contrast": contrast, "coef": b[k],
                         "se_firm": r_f.std_errors[term], "p_firm": r_f.pvalues[term],
                         "se_firm_sector_year": r_2.std_errors[term], "p_firm_sector_year": r_2.pvalues[term],
                         "se_sector": se_s, "p_sector_t8": 2 * stats.t.sf(abs(b[k] / se_s), G - 1),
                         "p_wcr_webb": p_webb, "webb_vectors": n_webb, "p_wcr_rademacher": p_rad, "rademacher_vectors": n_rad,
                         "n": n, "firms": dd.firm.nunique(), "sectors": G, "sector_years": dd.sector_year.nunique(),
                         "firms_switching_sector": switchers})
            notes.append(f"{hyp} {RES[res]:20s} {fe:8s} {term:20s} t(sector, own) = {t_obs:+.3f}  t(sector, linearmodels) = {b[k] / se_s:+.3f}  "
                         f"firms switching sector: {switchers}")
    R = pd.DataFrame(rows)
    R.to_csv(core.out("sector_inference.csv"), index=False)
    show = R[["hypothesis", "resource", "term", "contrast", "design", "coef", "se_firm", "p_firm", "se_firm_sector_year", "p_firm_sector_year",
              "se_sector", "p_sector_t8", "p_wcr_webb", "p_wcr_rademacher", "n", "sector_years"]]
    txt = ("Key estimates under sector-level inference (y = ln(1 + C))\n" + show.round(4).to_string(index=False) +
           "\n\nChecks: coefficients and two-way SEs equal the main model estimates; the Frisch-Waugh-Lovell replica equals PanelOLS;"
           "\nthe bootstrap at v = 1 reproduces the observed coefficient and cluster scores.\n" + "\n".join(notes) + "\n")
    core.out("sector_inference.txt").write_text(txt, encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
