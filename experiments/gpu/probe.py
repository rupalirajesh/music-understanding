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


def load_xy(acts_dir: Path, man: pd.DataFrame, target: str, layer: int):
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
        groups.append(factors.get("soundfont", "synth"))
    return np.array(X), np.array(y), np.array(groups)


def probe_layer(X, y, groups) -> float:
    """Accuracy averaged over held-out-soundfont folds."""
    accs = []
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
         manifest="manifests/stimuli.parquet", max_layers=40):
    man = pd.read_parquet(manifest)
    man = man[man["task"] == task].reset_index(drop=True)
    acts = Path(acts_dir)
    print(f"{task} / target={target}: {len(man)} stimuli")
    results = []
    for layer in range(max_layers):
        X, y, g = load_xy(acts, man, target, layer)
        if X is None or len(X) == 0:
            break
        acc = probe_layer(X, y, g)
        chance = 1 / len(np.unique(y))
        results.append({"layer": layer, "probe_acc": acc, "chance": chance})
        print(f"  layer {layer:2d}: {acc:.3f} (chance {chance:.3f})")
    pd.DataFrame(results).to_csv(f"probe__{task}__{target}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--target", required=True,
                    help="factor name (pitch_class, mode, quality, ...) or 'ground_truth'")
    ap.add_argument("--manifest", default="manifests/stimuli.parquet")
    args = ap.parse_args()
    main(args.acts, args.task, args.target, args.manifest)
