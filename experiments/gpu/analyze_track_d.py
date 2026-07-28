"""Conclusive Track-D analysis: does the spectrogram image GENUINELY help?

Reads responses__qwen25omni-lora-mix-s*.parquet (one per seed), scores each with
musicprobe.scoring (same parse/is_correct as everything else), and answers the
question with the statistics the first pass lacked:

  PRIMARY (is the effect real?):  image vs no_image is a WITHIN-MODEL, SAME-
  STIMULUS paired comparison (both conditions were in the training mix, so
  neither is out-of-distribution). Per task -> McNemar exact test on the
  discordant pairs, pooled over seeds, + a cluster-bootstrap 95% CI on Δacc.

  MECHANISM (why?):  wrong_image (does content matter?) and image_wrong_audio
  (SUBSTITUTE vs COMPLEMENT — does the image just carry the answer?).

  NEGATIVE CONTROL:  note_count — a spectrogram shouldn't help counting events.

  python gpu/analyze_track_d.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from musicprobe.config import RESULTS_DIR
from musicprobe.image_jobs import IMAGE_JOBS_PATH
from musicprobe.scoring import parse_response, is_correct

CONDS = ["no_image", "image", "wrong_image", "image_wrong_audio"]
CLABEL = {"no_image": "audio only", "image": "audio + image",
          "wrong_image": "audio + WRONG image", "image_wrong_audio": "WRONG audio + image"}
CCOLOR = {"no_image": "#9aa0a6", "image": "#0a9648",
          "wrong_image": "#e8710a", "image_wrong_audio": "#c0392b"}
TASK_ORDER = ["cents_discrimination", "tuning_judgment", "octave_id", "note_count"]
RNG = np.random.default_rng(0)


def _score_all():
    jobs = pd.read_parquet(IMAGE_JOBS_PATH)
    seeds = sorted(RESULTS_DIR.glob("responses__qwen25omni-lora-mix-s*.parquet"))
    frames = []
    for f in seeds:
        r = pd.read_parquet(f)[["job_id", "seed", "raw_response"]]
        d = r.merge(jobs, on="job_id", how="left")
        d = d[d.raw_response.notna()].copy()
        d["parsed"] = [parse_response(row) for row in d.itertuples()]
        d["correct"] = [is_correct(row.task, row.parsed, row.ground_truth)
                        for row in d.itertuples()]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["correct"] = df["correct"].eq(True)
    return df


def _paired(df, task, a="no_image", b="image"):
    """Paired (stimulus x seed) correctness for two conditions; McNemar + Δacc."""
    sub = df[df.task == task]
    wide = (sub[sub.image_condition.isin([a, b])]
            .pivot_table(index=["stimulus_id", "seed"], columns="image_condition",
                         values="correct", aggfunc="first").dropna())
    if a not in wide or b not in wide or len(wide) == 0:
        return None
    A, B = wide[a].astype(bool).values, wide[b].astype(bool).values
    n = len(A)
    b_only = int((B & ~A).sum())   # image right, audio-only wrong
    c_only = int((A & ~B).sum())   # audio-only right, image wrong
    dacc = (b_only - c_only) / n
    p = binomtest(min(b_only, c_only), b_only + c_only, 0.5).pvalue if (b_only + c_only) else 1.0
    # cluster bootstrap CI on Δacc (resample stimuli)
    stims = wide.reset_index()["stimulus_id"].values
    uniq = np.unique(stims)
    diffs = (B.astype(int) - A.astype(int))
    boots = []
    for _ in range(2000):
        pick = RNG.choice(uniq, len(uniq), replace=True)
        mask = np.isin(stims, pick)
        boots.append(diffs[mask].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(task=task, n=n, acc_audio=A.mean(), acc_image=B.mean(),
                dacc=dacc, ci_lo=lo, ci_hi=hi, mcnemar_p=p,
                b_only=b_only, c_only=c_only)


def _acc(df, task, cond):
    s = df[(df.task == task) & (df.image_condition == cond)]["correct"]
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    vals = s.astype(int).values
    boots = [RNG.choice(vals, len(vals), replace=True).mean() for _ in range(2000)]
    return vals.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    df = _score_all()
    n_seeds = df.seed.nunique()
    tasks = [t for t in TASK_ORDER if t in df.task.unique()]
    print(f"seeds={n_seeds}  tasks={tasks}\n")

    prim = pd.DataFrame([r for r in (_paired(df, t) for t in tasks) if r])
    print("=== PRIMARY: audio+image vs audio-only (paired, pooled over seeds) ===")
    print(prim.assign(**{
        "audio": prim.acc_audio.round(3), "img": prim.acc_image.round(3),
        "Δacc": prim.dacc.round(3), "95%CI": prim.apply(lambda r: f"[{r.ci_lo:+.2f},{r.ci_hi:+.2f}]", axis=1),
        "McNemar_p": prim.mcnemar_p.round(4)})[
        ["task", "n", "audio", "img", "Δacc", "95%CI", "McNemar_p"]].to_string(index=False))

    # figure: grouped bars per task, 4 conditions, bootstrap CIs, McNemar star
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(tasks)); w = 0.2
    for i, c in enumerate(CONDS):
        accs, los, his = zip(*[_acc(df, t, c) for t in tasks])
        accs = np.array(accs)
        err = np.array([accs - np.array(los), np.array(his) - accs])
        ax.bar(x + (i - 1.5) * w, accs, w, label=CLABEL[c], color=CCOLOR[c],
               yerr=err, capsize=3, error_kw=dict(lw=1, alpha=.6))
    # McNemar significance star over the image bar
    pmap = {r.task: r.mcnemar_p for r in prim.itertuples()}
    dmap = {r.task: r.dacc for r in prim.itertuples()}
    for j, t in enumerate(tasks):
        p = pmap.get(t, 1.0)
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
        ax.text(x[j] - 0.5 * w, 1.02, f"Δ={dmap[t]:+.2f}\n{star}", ha="center",
                fontsize=9, fontweight="bold",
                color="#0a5c2e" if (p < .05 and dmap[t] > 0) else "#555")
    ax.axhline(0.5, ls=":", c="gray", alpha=.6)
    ax.set_xticks(x); ax.set_xticklabels([t.replace("_", "\n") for t in tasks])
    ax.set_ylabel("accuracy (bootstrap 95% CI)"); ax.set_ylim(0, 1.15)
    ax.set_title(f"Track D (conclusive): does a spectrogram image genuinely help?\n"
                 f"within-model paired eval, {n_seeds} seeds, McNemar on image-vs-audio-only",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2, loc="upper right"); ax.grid(alpha=.2, axis="y")
    out = RESULTS_DIR / "trackd_conclusive_graph.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    prim.to_csv(RESULTS_DIR / "trackd_conclusive_summary.csv", index=False)
    print(f"\nwrote {out}\nwrote {RESULTS_DIR/'trackd_conclusive_summary.csv'}")


if __name__ == "__main__":
    main()
