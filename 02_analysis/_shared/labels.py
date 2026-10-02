import numpy as np
import codebook as cb

SPEC_MIN = 3
KEYS = [k for k, _ in cb.DETAIL_DEFS]

def vote(df):
    def f(r):
        v = r.dropna()
        if not len(v): return np.nan
        c = v.value_counts()
        return c.index[0] if (c.iloc[0] >= 2 or len(v) == 1) else "NO_MAJORITY"
    return df.apply(f, axis=1)
