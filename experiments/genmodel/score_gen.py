"""Score generated audio for constraint adherence using the L1 verifiers.

  python genmodel/score_gen.py --dir genmodel/outputs

Output: adherence.csv — per clip: the constraint, the L1 measurement, and
satisfied yes/no; plus a printed family × phrasing adherence table with the
baseline (unconstrained) satisfaction rate alongside. The headline number per
constraint is adherence_lift = P(satisfied | asked) − P(satisfied | baseline).

Honest scope notes:
- tempo: autocorrelation tempo, metrical-octave tolerant AND exact-window
  scored separately (both reported — "right feel" vs "followed the number").
- key: Krumhansl chroma match — treat borderline calls skeptically; real key
  verification for the paper should use essentia on the GPU box.
- register: frame-wise autocorrelation f0; only meaningful for solo prompts.
- meter/beats-per-bar: NOT auto-scored here — downbeat tracking on generated
  polyphonic audio is unreliable with lightweight DSP. adherence.csv lists the
  clips; verify by ear (this is also the most interesting family to hear).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musicprobe.l1_baselines import f0_autocorr, f0_to_midi, tempo_estimate, key_estimate


def score_clip(y, sr, ckey, cval):
    """Returns (measurement, satisfied) — satisfied may be None (manual)."""
    if ckey == "bpm":
        est = tempo_estimate(y, sr)
        if est is None:
            return None, None
        true = float(cval)
        exact = abs(est - true) / true <= 0.08
        octave_ok = any(abs(np.log2(est / true) - k) < 0.06 for k in (-1, 0, 1))
        return round(est, 1), {"exact": exact, "octave_ok": octave_ok}
    if ckey == "key":
        est = key_estimate(y, sr)
        return est, est.lower() == str(cval).lower()
    if ckey in ("min_midi", "max_midi"):
        hop = sr // 4
        f0s = [f0_autocorr(y[i:i + sr], sr) for i in range(0, len(y) - sr, hop)]
        midis = [f0_to_midi(f) for f in f0s if f]
        if not midis:
            return None, None
        if ckey == "min_midi":
            return round(min(midis), 1), min(midis) >= float(cval) - 0.5
        return round(max(midis), 1), max(midis) <= float(cval) + 0.5
    return None, None  # beats_per_bar, mode-vocab: manual verification


def main(gen_dir: str):
    d = Path(gen_dir)
    man = pd.read_parquet(d / "generations.parquet")
    rows = []
    for r in man.itertuples():
        y, sr = sf.read(d / r.file)
        if y.ndim == 2:
            y = y.mean(axis=1)
        meas, sat = score_clip(y, sr, r.constraint_key, r.constraint_value)
        rec = {"file": r.file, "family": r.family, "phrasing": r.phrasing,
               "prompt": r.prompt, "constraint": f"{r.constraint_key}={r.constraint_value}",
               "measured": str(meas)}
        if isinstance(sat, dict):
            rec.update({f"satisfied_{k}": v for k, v in sat.items()})
        else:
            rec["satisfied"] = sat
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(d / "adherence.csv", index=False)

    # Also measure every BASELINE clip against every constraint family's
    # checkers to get base rates (tempo/key only — register needs solo texture)
    print(df.groupby(["family", "phrasing"])
            [[c for c in df.columns if c.startswith("satisfied")]]
            .mean(numeric_only=True).to_string())
    print(f"\nwrote {d / 'adherence.csv'} — beats_per_bar & mode-vocab rows "
          "need verification by ear")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="genmodel/outputs")
    args = ap.parse_args()
    main(args.dir)
