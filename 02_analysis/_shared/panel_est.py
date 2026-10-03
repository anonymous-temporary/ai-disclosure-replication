import pandas as pd
from linearmodels.panel import PanelOLS

CTRL = ["L1_SIZE", "L1_LEV_w", "L1_CASH_AT_w", "L1_CAPEX_AT_w", "L1_ROA_w", "LOG_WORDS"]
RD = ["L1_RD_SALES0", "L1_RD_MISSING", "L1_PRE_REVENUE"]
ROWS, TXT = [], []

def est(d, y, x, fe, label, show=None, group="cik"):
    cols = [y] + x + ["cik", "fy", "ind_year"]
    dd = d.dropna(subset=[c for c in cols if c in d]).copy()
    x = [v for v in x if dd[v].nunique() > 1]
    if len(dd) < 200 or not x:
        TXT.append(f"\n--- {label} [{fe}]  y = {y}: skipped (N = {len(dd)}, {len(x)} regressors with variation)\n"); return None
    dd["firm"] = dd["cik"]
    dd["sector_year"] = pd.factorize(dd["industry"].astype(str) + "_" + dd["fy"].astype(int).astype(str))[0]
    dd = dd.set_index([group, "fy_idx" if "fy_idx" in dd else "fy"])
    try:
        if fe == "WITHIN":
            mod = PanelOLS(dd[y], dd[x], entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=True)
        else:
            mod = PanelOLS(dd[y], dd[x], other_effects=dd[["ind_year"]].astype("category"), drop_absorbed=True, check_rank=True)
        r = mod.fit(cov_type="clustered", clusters=dd[["firm", "sector_year"]])
    except Exception as e:
        TXT.append(f"\n--- {label} [{fe}]  y = {y}: FAILED {type(e).__name__}\n"); return None
    show = show or x
    tab = pd.DataFrame({"coef": r.params, "se": r.std_errors, "t": r.tstats, "p": r.pvalues}).loc[[s for s in show if s in r.params.index]].round(4)
    TXT.append(f"\n--- {label} [{fe}]  y = {y}  N = {int(r.nobs):,}  firms = {dd['firm'].nunique():,}  R2(within) = {r.rsquared:.3f}\n{tab.to_string()}\n")
    for k, row in tab.iterrows():
        ROWS.append({"model": label, "fe": fe, "y": y, "term": k, **row.to_dict(), "n": int(r.nobs)})
    return r
