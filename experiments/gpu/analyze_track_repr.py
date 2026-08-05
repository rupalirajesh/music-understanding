"""Analysis for Tracks L-W (registry-based, parallels gpu/analyze_track_g.py's
statistics exactly -- see that file's docstring for the full method
explanation: PRIMARY image-vs-no_image paired McNemar + cluster-bootstrap CI,
MECHANISM wrong_image/image_wrong_audio controls). One script instead of
twelve near-duplicates, same DRY reasoning as train_track_repr.py.

  python gpu/analyze_track_repr.py --track L
  python gpu/analyze_track_repr.py --track L --compare M N O P Q   # side-by-side summary
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
from musicprobe.config import RESULTS_DIR, MANIFEST_DIR  # noqa: E402
from musicprobe.scoring import parse_response, is_correct  # noqa: E402
from musicprobe.paired_eval import paired_delta, bootstrap_acc, star  # noqa: E402
from train_track_repr import TRACKS, HARMONY_TASKS, RHYTHM_TASKS  # noqa: E402

CONDS = ["no_image", "image", "wrong_image", "image_wrong_audio"]
TASK_ORDER = {**{t: HARMONY_TASKS for t in HARMONY_TASKS}, **{t: RHYTHM_TASKS for t in RHYTHM_TASKS}}


def _score_all(track: str, tag: str):
    spec = TRACKS[track]
    jobs_path = MANIFEST_DIR / f"{spec['name'].replace('-', '_')}_jobs.parquet"
    jobs = pd.read_parquet(jobs_path)
    pattern = f"responses__{tag}-s*.parquet"
    seeds = sorted(RESULTS_DIR.glob(pattern))
    if not seeds:
        raise FileNotFoundError(f"no response files matching {pattern} in {RESULTS_DIR} -- "
                                f"run gpu/train_track_repr.py --track {track} on the H100 box first.")
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
    return dict(task=r["task"], n=r["n"], acc_a=r["acc_a"], acc_b=r["acc_b"],
               dacc=r["dacc"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], mcnemar_p=r["mcnemar_p"],
               b_only=r["b_only"], c_only=r["c_only"])


def analyze_one(track: str):
    spec = TRACKS[track]
    tag = f"qwen25omni-{spec['name']}"
    tasks = spec["tasks"]
    df = _score_all(track, tag)
    n_seeds = df.seed.nunique()
    print(f"=== Track {track} ({spec['name']}): seeds={n_seeds} tasks={list(tasks)} ===")

    prim = pd.DataFrame([r for r in (_paired(df, t) for t in tasks) if r])
    print(prim.assign(**{
        "audio": prim.acc_a.round(3), "img": prim.acc_b.round(3),
        "Δacc": prim.dacc.round(3),
        "95%CI": prim.apply(lambda r: f"[{r.ci_lo:+.2f},{r.ci_hi:+.2f}]", axis=1),
        "McNemar_p": prim.mcnemar_p.round(4)})[
        ["task", "n", "audio", "img", "Δacc", "95%CI", "McNemar_p"]].to_string(index=False))

    mech_wrong_img = pd.DataFrame(
        [r for r in (_paired(df, t, "no_image", "wrong_image") for t in tasks) if r])
    mech_wrong_audio = pd.DataFrame(
        [r for r in (_paired(df, t, "image", "image_wrong_audio") for t in tasks) if r])
    print("\n-- MECHANISM: no_image vs wrong_image (content matter?) --")
    print(mech_wrong_img[["task", "dacc", "mcnemar_p"]].to_string(index=False) if len(mech_wrong_img) else "n/a")
    print("-- MECHANISM: image vs image_wrong_audio (substitute vs complement?) --")
    print(mech_wrong_audio[["task", "dacc", "mcnemar_p"]].to_string(index=False) if len(mech_wrong_audio) else "n/a")

    fig, ax = plt.subplots(figsize=(3 * len(tasks) + 2, 6))
    x = np.arange(len(tasks)); w = 0.2
    for i, c in enumerate(CONDS):
        accs, los, his = [], [], []
        for t in tasks:
            a, lo, hi = bootstrap_acc(df, t, "image_condition", c)
            accs.append(a); los.append(lo); his.append(hi)
        accs = np.array(accs)
        err = np.array([accs - np.array(los), np.array(his) - accs])
        ax.bar(x + (i - 1.5) * w, accs, w, label=c, yerr=err, capsize=3,
              error_kw=dict(lw=1, alpha=.6))
    ax.set_xticks(x); ax.set_xticklabels([t.replace("_", "\n") for t in tasks])
    ax.set_ylabel("accuracy (bootstrap 95% CI)")
    ax.set_title(f"Track {track}: {spec['name']}\nwithin-model paired eval, {n_seeds} seeds",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.2, axis="y")
    out_png = RESULTS_DIR / f"track{track.lower()}_{spec['name'].replace('-', '_')}_graph.png"
    out_csv = RESULTS_DIR / f"track{track.lower()}_{spec['name'].replace('-', '_')}_summary.csv"
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight")
    prim.rename(columns={"acc_a": "acc_no_image", "acc_b": "acc_image"}).to_csv(out_csv, index=False)
    print(f"\nwrote {out_png}\nwrote {out_csv}")
    return prim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--compare", nargs="*", default=None,
                    help="also analyze these tracks and print a side-by-side "
                         "best-Δacc-per-task comparison (only tracks with response "
                         "files already present will succeed)")
    args = ap.parse_args()

    results = {args.track: analyze_one(args.track)}
    if args.compare:
        for t in args.compare:
            try:
                results[t] = analyze_one(t)
            except FileNotFoundError as e:
                print(f"[skip Track {t}] {e}")

    if len(results) > 1:
        print("\n=== Cross-track comparison (Δacc, image vs no_image) ===")
        rows = []
        for t, df in results.items():
            for r in df.itertuples():
                rows.append({"track": t, "task": r.task, "dacc": r.dacc, "p": r.mcnemar_p})
        cmp = pd.DataFrame(rows).pivot(index="task", columns="track", values="dacc")
        print(cmp.round(3).to_string())


if __name__ == "__main__":
    main()
