"""Track B next action 19 -- nonlinear decoder per encoder layer, the same
saved-activation probing as probe.py but with a small MLP instead of
LogisticRegression. Every existing probe in this project (probe.py,
probe_microtone.py, probe_vision_pitch.py) is a LINEAR classifier -- if a
straight line can't separate the classes, the standing conclusion has been
"the information isn't really in the representation." A nonlinear decoder
tests whether that conclusion was an artifact of the tool (linear-
inseparable but present) rather than the representation (genuinely absent).
Most relevant for the near-floor tasks flagged in PROJECT_STATE.md next
action 13: mode_id (best linear probe 0.04-0.12 vs chance 0.077, barely
above chance on EVERY encoder) and, to a lesser extent, interval_id/
chord_quality.

New file, not a change to probe.py -- same reasoning as image_track_common.py
leaving train_track_g_chroma.py untouched: probe.py's own CSV outputs are
already cited in PAPER.md/PROJECT_STATE.md, so its behavior must not change
underneath already-reported numbers. This mirrors probe.py's interface
exactly (same --acts/--task/--target args, same held-out-soundfont fold
discipline, same output naming convention with an "mlp" tag instead of the
encoder dir name alone) so it's a drop-in second pass over the same
activations, not a new pipeline.

  python probe_mlp.py --acts /path/to/acts --target pitch_class --task pitch_note_id

UNVERIFIED end-to-end: this laptop has no copy of the raw activation .npz
files (musicprobe/config.py: "activations stay on the GPU box") -- only
probe.py's already-run CSV *outputs* are committed to this repo. Logic
mirrors probe.py's load_xy/probe_layer exactly (same file layout, same
target/group extraction), which IS the already-verified part; only the
classifier swap (LogisticRegression -> MLPClassifier) is new and untested
against real data. Run on the H100 box against an existing --acts directory
before trusting the numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from probe import load_xy  # noqa: E402 -- identical activation-loading logic, reused not copied

HIDDEN = (32,)     # small on purpose: with ~100-200 examples/fold (see PROJECT_STATE
                    # task-size table), a bigger net would just memorize -- this is
                    # meant to catch simple nonlinear separability, not replace L3
                    # behavioral evidence with a black box.


def probe_layer_mlp(X, y, groups, seed=0) -> float:
    """Same held-out-soundfont-fold discipline as probe.probe_layer, MLP
    instead of logistic regression."""
    def _clf():
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=HIDDEN, activation="relu",
                                           alpha=1e-2, max_iter=2000, random_state=seed,
                                           early_stopping=True, n_iter_no_change=10))
    accs = []
    if len(np.unique(groups)) < 2:
        idx = np.random.default_rng(seed).permutation(len(y))
        cut = int(0.75 * len(y))
        tr_i, te_i = idx[:cut], idx[cut:]
        if len(np.unique(y[tr_i])) < 2 or len(te_i) < 5:
            return np.nan
        clf = _clf(); clf.fit(X[tr_i], y[tr_i])
        return float(clf.score(X[te_i], y[te_i]))
    for held in np.unique(groups):
        tr, te = groups != held, groups == held
        if len(np.unique(y[tr])) < 2 or te.sum() < 5:
            continue
        clf = _clf(); clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs)) if accs else np.nan


def main(acts_dir: str, task: str, target: str,
         manifest="manifests/stimuli.parquet", max_layers=40, seed=0,
         out_dir="results/trackB/probes", group_key: str = "soundfont"):
    man = pd.read_parquet(manifest)
    man = man[man["task"] == task].reset_index(drop=True)
    acts = Path(acts_dir)
    print(f"[mlp] {task} / target={target}: {len(man)} stimuli")
    results = []
    for layer in range(max_layers):
        X, y, g = load_xy(acts, man, target, layer, group_key)
        if X is None or len(X) == 0:
            break
        lin_acc = None  # not recomputed here -- cross-reference the existing
                         # probe__*.csv from probe.py for the linear number
        mlp_acc = probe_layer_mlp(X, y, g, seed=seed)
        chance = 1 / len(np.unique(y))
        results.append({"layer": layer, "probe_acc_mlp": mlp_acc, "chance": chance})
        print(f"  layer {layer:2d}: mlp={mlp_acc:.3f} (chance {chance:.3f})")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"probe_mlp__{acts.name}__{task}__{target}.csv"
    pd.DataFrame(results).to_csv(dst, index=False)
    print(f"[mlp] wrote {dst} -- diff against probe__{acts.name}__{task}__{target}.csv "
          f"(probe.py's linear result) to see whether nonlinearity recovers anything.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--target", required=True,
                    help="factor name (pitch_class, mode, quality, ...) or 'ground_truth'")
    ap.add_argument("--manifest", default="manifests/stimuli.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/trackB/probes")
    ap.add_argument("--group-key", default="soundfont",
                    help="factors.* key to hold out on -- 'track_id' for real_music_medleydb, "
                         "'instrument_family' for real_music_nsynth (next action 25)")
    args = ap.parse_args()
    main(args.acts, args.task, args.target, args.manifest, seed=args.seed, out_dir=args.out,
        group_key=args.group_key)
