import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_shared"))
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

import core
import panel_est as pe

warnings.filterwarnings("ignore")
RES = {"L1_RD_SALES0": "RE1", "L1_LOG_AI_PAT_STOCK": "RE2", "L1_AI_WORKER": "RE3"}


def controls(res):
    return [v for v in pe.RD if v != res] + pe.CTRL if res == "L1_RD_SALES0" else pe.RD + pe.CTRL


def fit(s, x, design):
    dd = s.dropna(subset=["ln_C"] + x + ["cik", "fy", "ind_year"]).copy(); x = [v for v in x if dd[v].nunique() > 1]
    dd["firm"] = dd.cik; dd["sector_year"] = pd.factorize(dd.ind_year)[0]; dd = dd.set_index(["cik", "fy"])
    other = dd[["ind_year"]].astype("category")
    mod = {"within": lambda: PanelOLS(dd.ln_C, dd[x], entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=True),
           "between": lambda: PanelOLS(dd.ln_C, dd[x], other_effects=other, drop_absorbed=True, check_rank=True),
           "firm_sector_year": lambda: PanelOLS(dd.ln_C, dd[x], entity_effects=True, other_effects=other, drop_absorbed=True, check_rank=True)}[design]()
    return mod.fit(cov_type="clustered", clusters=dd[["firm", "sector_year"]]), len(dd), dd.firm.nunique()


def main():
    m = pd.read_parquet(core.out("panel.parquet"))
    d = m[(m.is_operating == 1) & m.fy.between(2015, 2025) & (m.coded == 1)].copy()
    d["ind_year"] = d.industry + "_" + d.fy.astype(int).astype(str); d["ln_C"] = np.log1p(d.C); rows = []
    for res, lab in RES.items():
        s = (d[d.fy <= 2024] if "PAT" in res else d).copy(); g = s.industry.map(core.AI_MODE)
        for k, short in (("producer", "S1"), ("co-developer", "S2"), ("adopter", "S3")):
            s[f"M_{short}"] = (g == k).astype(float); s[f"Rx{short}"] = s[res] * s[f"M_{short}"]
        for design in ("within", "between", "firm_sector_year"):
            r, n, f = fit(s, [res] + controls(res), design)
            rows.append(dict(resource=lab, term="H1 slope", design=design, coef=r.params[res], se=r.std_errors[res], p=r.pvalues[res], n=n, firms=f))
            r, n, f = fit(s, [res, "M_S1", "M_S2", "RxS1", "RxS2"] + controls(res), design)
            for t, name in ((res, "S3 slope"), ("RxS1", "S1-S3"), ("RxS2", "S2-S3")):
                rows.append(dict(resource=lab, term=name, design=design, coef=r.params[t], se=r.std_errors[t], p=r.pvalues[t], n=n, firms=f))
    out = pd.DataFrame(rows)
    pub = pd.read_csv(core.out("models.csv")); pub = pub[(pub.y == "ln_C") & pub.term.isin(RES) & pub.table.isin(["T2", "T2V"])]
    for _, q in pub.iterrows():
        v = out[(out.resource == RES[q.term]) & (out.term == "H1 slope") & (out.design == q.fe.lower())].iloc[0]
        assert abs(v.coef - q.coef) < 1e-8 and v.n == q.n, (q.term, q.fe, v.coef, q.coef)
    dest = _Path(__file__).resolve().parent / "results"; dest.mkdir(exist_ok=True)
    out.to_csv(dest / "firm_sector_year_check.csv", index=False)
    print(out[out.design == "firm_sector_year"].round(4).to_string(index=False)); print("reproduction of the main models: OK,", len(pub), "rows")


if __name__ == "__main__":
    main()
