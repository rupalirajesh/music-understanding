"""Track H analysis: does an in-audio reference tone genuinely help
tuning_judgment (absolute tuning)?

Same statistics as Track D/G (musicprobe.paired_eval):
  PRIMARY:   reftone vs plain, within-model paired (both in the dropout
             training mix) -> McNemar exact test + cluster-bootstrap CI.
  MECHANISM: wrong_reftone vs plain — does a WRONG reference mislead the
             model (real target-vs-reference comparison) or match plain/
             reftone (just reacting to "two tones present", not comparing)?

  python gpu/analyze_track_h.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402
from musicprobe.config import RESULTS_DIR
from musicprobe.reftone_jobs import REFTONE_JOBS_PATH
from musicprobe.scoring import parse_response, is_correct
from musicprobe.paired_eval import paired_delta, bootstrap_acc, star

CONDS = ["plain", "reftone", "wrong_reftone"]
CLABEL = {"plain": "target tone alone", "reftone": "reference + target",
          "wrong_reftone": "WRONG reference + target"}
CCOLOR = {"plain": "#9aa0a6", "reftone": "#0a9648", "wrong_reftone": "#c0392b"}


def _score_all(pattern="responses__qwen25omni-reftone-s*.parquet"):
    jobs = pd.read_parquet(REFTONE_JOBS_PATH)
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


def _paired(df, task, a, b):
    return paired_delta(df, task, "reftone_condition", a, b)


def _acc(df, task, cond):
    return bootstrap_acc(df, task, "reftone_condition", cond)


def main():
    df = _score_all()
    n_seeds = df.seed.nunique()
    task = "tuning_judgment"
    print(f"seeds={n_seeds}  task={task}  n_jobs={len(df)}\n")

    help_r = _paired(df, task, "plain", "reftone")
    uses_r = _paired(df, task, "plain", "wrong_reftone")
    print("=== PRIMARY: reftone vs plain (paired, pooled over seeds) ===")
    if help_r:
        print(f"  plain={help_r['acc_a']:.3f}  reftone={help_r['acc_b']:.3f}  "
              f"Δacc={help_r['dacc']:+.3f}  95%CI=[{help_r['ci_lo']:+.2f},{help_r['ci_hi']:+.2f}]  "
              f"McNemar_p={help_r['mcnemar_p']:.4f} {star(help_r['mcnemar_p'])}")
    print("=== MECHANISM: wrong_reftone vs plain (does a WRONG reference mislead it?) ===")
    if uses_r:
        print(f"  plain={uses_r['acc_a']:.3f}  wrong_reftone={uses_r['acc_b']:.3f}  "
              f"Δacc={uses_r['dacc']:+.3f}  95%CI=[{uses_r['ci_lo']:+.2f},{uses_r['ci_hi']:+.2f}]  "
              f"McNemar_p={uses_r['mcnemar_p']:.4f} {star(uses_r['mcnemar_p'])}")
        print("  (a genuine comparison should mislead the model here, i.e. a negative")
        print("   swing vs reftone; if wrong_reftone ≈ reftone, the model isn't really")
        print("   comparing target-to-reference, just reacting to 'two tones present')")

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(1); w = 0.25
    for i, c in enumerate(CONDS):
        acc, lo, hi = _acc(df, task, c)
        err = np.array([[acc - lo], [hi - acc]])
        ax.bar(x + (i - 1) * w, [acc], w, label=CLABEL[c], color=CCOLOR[c],
               yerr=err, capsize=4, error_kw=dict(lw=1, alpha=.6))
    if help_r:
        ax.text(0, 1.03, f"help Δ={help_r['dacc']:+.2f} {star(help_r['mcnemar_p'])}",
                ha="center", fontsize=9, fontweight="bold",
                color="#0a5c2e" if (help_r["mcnemar_p"] < .05 and help_r["dacc"] > 0) else "#555")
    if uses_r:
        ax.text(0, 1.10, f"wrong-ref mislead? {star(uses_r['mcnemar_p'])} (Δ={uses_r['dacc']:+.2f})",
                ha="center", fontsize=8.5,
                color="#b35900" if uses_r["mcnemar_p"] < .05 else "#999")
    ax.axhline(0.5, ls=":", c="gray", alpha=.6)
    ax.set_xticks(x); ax.set_xticklabels(["tuning_judgment"])
    ax.set_ylabel("accuracy (bootstrap 95% CI)"); ax.set_ylim(0, 1.2)
    ax.set_title("Track H: does an in-AUDIO reference tone fix absolute tuning?\n"
                 f"within-model paired eval, {n_seeds} seeds, McNemar on reftone-vs-plain",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=.2, axis="y")
    out = RESULTS_DIR / "trackh_reftone_graph.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    pd.DataFrame([r for r in (help_r, uses_r) if r]).to_csv(
        RESULTS_DIR / "trackh_reftone_summary.csv", index=False)
    print(f"\nwrote {out}\nwrote {RESULTS_DIR/'trackh_reftone_summary.csv'}")


if __name__ == "__main__":
    main()
