"""Track G analysis: does the chromagram image GENUINELY help the harmonic
task cluster (key_id/mode_id/chord_quality/interval_id)?

Same statistics as Track D's conclusive/force analysis (musicprobe.paired_eval):
  PRIMARY:   image vs no_image, within-model paired (both in the modality-
             dropout training mix) -> McNemar exact test + cluster-bootstrap CI.
  MECHANISM: wrong_image (does content matter, not just presence?) and
             image_wrong_audio (substitute vs complement).

No spare negative-control task this round — all 4 tasks here are the
hypothesis-relevant set (Track B's L3>L2 harmonic shortlist); unlike Track D,
there's no already-solved sibling task in this cluster to use as a null check.

  python gpu/analyze_track_g.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402
from musicprobe.config import RESULTS_DIR, MANIFEST_DIR
from musicprobe.scoring import parse_response, is_correct
from musicprobe.paired_eval import paired_delta, bootstrap_acc, star

CHROMA_JOBS_PATH = MANIFEST_DIR / "chroma_jobs.parquet"
CONDS = ["no_image", "image", "wrong_image", "image_wrong_audio"]
CLABEL = {"no_image": "audio only", "image": "audio + chromagram",
          "wrong_image": "audio + WRONG chromagram", "image_wrong_audio": "WRONG audio + chromagram"}
CCOLOR = {"no_image": "#9aa0a6", "image": "#0a9648",
          "wrong_image": "#e8710a", "image_wrong_audio": "#c0392b"}
TASK_ORDER = ["key_id", "mode_id", "chord_quality", "interval_id"]


def _score_all(pattern="responses__qwen25omni-chroma-s*.parquet"):
    jobs = pd.read_parquet(CHROMA_JOBS_PATH)
    seeds = sorted(RESULTS_DIR.glob(pattern))
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
    r = paired_delta(df, task, "image_condition", a, b)
    if r is None:
        return None
    return dict(task=r["task"], n=r["n"], acc_audio=r["acc_a"], acc_image=r["acc_b"],
                dacc=r["dacc"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], mcnemar_p=r["mcnemar_p"],
                b_only=r["b_only"], c_only=r["c_only"])


def _acc(df, task, cond):
    return bootstrap_acc(df, task, "image_condition", cond)


def main():
    df = _score_all()
    n_seeds = df.seed.nunique()
    tasks = [t for t in TASK_ORDER if t in df.task.unique()]
    print(f"seeds={n_seeds}  tasks={tasks}\n")

    prim = pd.DataFrame([r for r in (_paired(df, t) for t in tasks) if r])
    print("=== PRIMARY: audio+chromagram vs audio-only (paired, pooled over seeds) ===")
    print(prim.assign(**{
        "audio": prim.acc_audio.round(3), "img": prim.acc_image.round(3),
        "Δacc": prim.dacc.round(3), "95%CI": prim.apply(lambda r: f"[{r.ci_lo:+.2f},{r.ci_hi:+.2f}]", axis=1),
        "McNemar_p": prim.mcnemar_p.round(4)})[
        ["task", "n", "audio", "img", "Δacc", "95%CI", "McNemar_p"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(tasks)); w = 0.2
    for i, c in enumerate(CONDS):
        accs, los, his = zip(*[_acc(df, t, c) for t in tasks])
        accs = np.array(accs)
        err = np.array([accs - np.array(los), np.array(his) - accs])
        ax.bar(x + (i - 1.5) * w, accs, w, label=CLABEL[c], color=CCOLOR[c],
               yerr=err, capsize=3, error_kw=dict(lw=1, alpha=.6))
    pmap = {r.task: r.mcnemar_p for r in prim.itertuples()}
    dmap = {r.task: r.dacc for r in prim.itertuples()}
    uses = {r["task"]: r for r in (_paired(df, t, "no_image", "wrong_image") for t in tasks) if r}
    for j, t in enumerate(tasks):
        p = pmap.get(t, 1.0)
        ax.text(x[j] - 0.5 * w, 1.02, f"help Δ={dmap[t]:+.2f} {star(p)}", ha="center",
                fontsize=8.5, fontweight="bold",
                color="#0a5c2e" if (p < .05 and dmap[t] > 0) else "#555")
        u = uses.get(t)
        if u:
            up = u["mcnemar_p"]
            ax.text(x[j] - 0.5 * w, 1.09, f"uses img? {star(up)} (wrong Δ={u['dacc']:+.2f})",
                    ha="center", fontsize=8,
                    color="#b35900" if up < .05 else "#999")
    ax.axhline(0.25, ls=":", c="gray", alpha=.6)  # 4-way MCQ chance, not 0.5 (Track D's tasks)
    ax.set_xticks(x); ax.set_xticklabels([t.replace("_", "\n") for t in tasks])
    ax.set_ylabel("accuracy (bootstrap 95% CI)"); ax.set_ylim(0, 1.15)
    ax.set_title("Track G: does a chromagram genuinely help the harmonic task cluster?\n"
                 f"within-model paired eval, {n_seeds} seeds, McNemar on chromagram-vs-audio-only",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2, loc="upper right"); ax.grid(alpha=.2, axis="y")
    out = RESULTS_DIR / "trackg_chroma_graph.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    prim.to_csv(RESULTS_DIR / "trackg_chroma_summary.csv", index=False)
    print(f"\nwrote {out}\nwrote {RESULTS_DIR/'trackg_chroma_summary.csv'}")


if __name__ == "__main__":
    main()
