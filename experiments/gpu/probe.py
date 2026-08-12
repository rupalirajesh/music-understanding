"""Track B step 2 (runs anywhere, CPU is fine): linear probes on saved
activations -> probe-accuracy-by-layer curves per musical property.

  python probe.py --acts /path/to/acts --target pitch_class --task pitch_note_id

Leakage guard: splits are BY SOUNDFONT (train on 2, test on the held-out one),
never by clip — a probe that only works within-soundfont is reading timbre
fingerprints, not music.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def load_xy(acts_dir: Path, man: pd.DataFrame, target: str, layer: int, group_key: str = "soundfont"):
    """group_key (added PROJECT_STATE next action 25, default unchanged =
    'soundfont' -- every already-run/verified probe call keeps its exact
    original behavior): which factors.* key to hold out on. The real-music
    manifests (real_music_medleydb.py, real_music_nsynth.py) have no
    soundfont, since there's no synthesized instrument to name -- they carry
    'track_id' or 'instrument_family' instead, same leakage-guard role
    (never let a probe see the same real recording/instrument in both train
    and held folds)."""
    X, y, groups = [], [], []
    for row in man.itertuples():
        f = acts_dir / (row.stimulus_id.replace("/", "__") + ".npz")
        if not f.exists():
            continue
        with np.load(f) as z:
            key = f"pooled_{layer:02d}"
            if key not in z:
                return None, None, None
            X.append(z[key])
        factors = row.factors if isinstance(row.factors, dict) else json.loads(row.factors)
        y.append(factors.get(target, row.ground_truth))
        groups.append(factors.get(group_key, "synth"))
    return np.array(X), np.array(y), np.array(groups)


def probe_layer(X, y, groups) -> float:
    """Accuracy averaged over held-out-soundfont folds."""
    accs = []
    if len(np.unique(groups)) < 2:
        # numpy-tone tasks (cents, tuning) have no soundfont factor -> no
        # leakage-guard folds possible; fall back to a seeded random split
        idx = np.random.default_rng(0).permutation(len(y))
        cut = int(0.75 * len(y))
        tr_i, te_i = idx[:cut], idx[cut:]
        if len(np.unique(y[tr_i])) < 2 or len(te_i) < 5:
            return np.nan
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr_i], y[tr_i])
        return float(clf.score(X[te_i], y[te_i]))
    for held in np.unique(groups):
        tr, te = groups != held, groups == held
        if len(np.unique(y[tr])) < 2 or te.sum() < 5:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs)) if accs else np.nan


def main(acts_dir: str, task: str, target: str,
         manifest="manifests/stimuli.parquet", max_layers=40,
         out_dir="results/trackB/probes", group_key: str = "soundfont"):
    man = pd.read_parquet(manifest)
    man = man[man["task"] == task].reset_index(drop=True)
    acts = Path(acts_dir)
    print(f"{task} / target={target}: {len(man)} stimuli")
    results = []
    for layer in range(max_layers):
        X, y, g = load_xy(acts, man, target, layer, group_key)
        if X is None or len(X) == 0:
            break
        acc = probe_layer(X, y, g)
        chance = 1 / len(np.unique(y))
        results.append({"layer": layer, "probe_acc": acc, "chance": chance})
        print(f"  layer {layer:2d}: {acc:.3f} (chance {chance:.3f})")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"probe__{acts.name}__{task}__{target}.csv"  # acts dir name = encoder tag
    pd.DataFrame(results).to_csv(dst, index=False)
    print(f"wrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--target", required=True,
                    help="factor name (pitch_class, mode, quality, ...) or 'ground_truth'")
    ap.add_argument("--manifest", default="manifests/stimuli.parquet")
    ap.add_argument("--out", default="results/trackB/probes")
    ap.add_argument("--group-key", default="soundfont",
                    help="factors.* key to hold out on -- 'track_id' for real_music_medleydb, "
                         "'instrument_family' for real_music_nsynth (next action 25)")
    args = ap.parse_args()
    main(args.acts, args.task, args.target, args.manifest, out_dir=args.out,
         group_key=args.group_key)
