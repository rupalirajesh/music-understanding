"""Shared statistics for within-model, same-stimulus paired eval comparisons
(Track D/E/F's methodology, factored out so Track G/H don't duplicate it).

The pattern, established in Track D's conclusive rerun (train_track_d_conclusive.py,
commit d7d4c3f): two conditions (e.g. no_image vs image, or plain vs reftone) that
were BOTH in the training distribution, evaluated on the same held-out stimuli x
seeds -> McNemar exact test on the discordant pairs (pooled over seeds) + a
cluster-bootstrap 95% CI on Delta-accuracy (resampling stimulus_id, the correct
clustering unit since multiple seeds share the same stimulus).
"""
import numpy as np
from scipy.stats import binomtest

RNG = np.random.default_rng(0)


def paired_delta(df, task: str, cond_col: str, a: str, b: str,
                  stimulus_col: str = "stimulus_id", seed_col: str = "seed",
                  n_boot: int = 2000) -> dict | None:
    """Paired (stimulus x seed) correctness for conditions a vs b under cond_col.
    Returns None if either condition has no rows for this task."""
    sub = df[df.task == task]
    wide = (sub[sub[cond_col].isin([a, b])]
            .pivot_table(index=[stimulus_col, seed_col], columns=cond_col,
                         values="correct", aggfunc="first").dropna())
    if a not in wide or b not in wide or len(wide) == 0:
        return None
    A, B = wide[a].astype(bool).values, wide[b].astype(bool).values
    n = len(A)
    b_only = int((B & ~A).sum())   # b right, a wrong
    c_only = int((A & ~B).sum())   # a right, b wrong
    dacc = (b_only - c_only) / n
    p = binomtest(min(b_only, c_only), b_only + c_only, 0.5).pvalue if (b_only + c_only) else 1.0
    stims = wide.reset_index()[stimulus_col].values
    uniq = np.unique(stims)
    diffs = B.astype(int) - A.astype(int)
    boots = []
    for _ in range(n_boot):
        pick = RNG.choice(uniq, len(uniq), replace=True)
        mask = np.isin(stims, pick)
        boots.append(diffs[mask].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(task=task, n=n, acc_a=A.mean(), acc_b=B.mean(),
                dacc=dacc, ci_lo=lo, ci_hi=hi, mcnemar_p=p,
                b_only=b_only, c_only=c_only)


def bootstrap_acc(df, task: str, cond_col: str, cond: str, n_boot: int = 2000):
    """Unpaired accuracy + bootstrap 95% CI for one (task, condition) cell."""
    s = df[(df.task == task) & (df[cond_col] == cond)]["correct"]
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    vals = s.astype(int).values
    boots = [RNG.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
    return vals.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def star(p: float) -> str:
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
