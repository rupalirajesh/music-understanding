"""Rigorous linear probe for MICROTONAL pitch — localizes where cent-level pitch
information survives in each frozen audio encoder (L2), so we know whether the
fix belongs at the encoder (pitch-aware front-end) or the LM readout (L3).

This replaces the generic gpu/probe.py for the two microtone tasks, whose design
made the result uninformative:
  * cents_discrimination is a tone PAIR; mean-pooling the whole clip averages the
    two tones together and destroys the (5-100 cent) delta.
  * base pitch is randomized over ~2 octaves, so a clip-pooled vector encodes
    absolute pitch, not the tiny offset.
  * numpy tones carry no soundfont, so the old code silently fell back to a single
    75/25 split — no CV, no CI, no permutation null.

Correct framings here:

  cents_discrimination  -> DIRECTION {higher,lower} decoded from the DIFFERENCE of
    per-tone pooled activations  d = pool(tone2) - pool(tone1).  Segment-aware
    pooling (tone1 = [0,1.0]s, gap [1.0,1.5]s, tone2 = [1.5,2.5]s) isolates the
    pitch CHANGE and cancels shared base-pitch + timbre.  Headline readout: probe
    accuracy vs |delta_cents| = a REPRESENTATIONAL PSYCHOMETRIC CURVE.

  tuning_judgment       -> SIGNED DETUNE (cents) ridge-regressed on the pooled
    single-tone activation.  Metric MAE(cents) & R^2, compared to the 12-TET-SNAP
    baseline (a pitch-quantized rep predicts 0 detune -> MAE = mean|detune|).
    Beating snap-to-grid == microtonal info survives; tying it == quantized rep.

Leakage guard for numpy tones: GroupKFold holds out whole BASE-PITCH bins (never
split by clip), so the probe must read the offset on pitches it never trained on.
Significance: label-permutation null (empirical p) + fold-wise spread.  A signal-
floor oracle decodes from the known f0s to confirm L1 is intact.

  python gpu/probe_microtone.py --acts acts --encoders whisper mert330 qwen25omni_own af3_own musicflamingo_own
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# tone_pair layout (musicprobe/synth.py): tone_dur=1.0s, gap=0.5s -> tone2 at 1.5s
TONE1 = (0.00, 1.00)
TONE2 = (1.50, 2.50)
CLIP_SECONDS = 2.50           # signal duration of a cents clip
DELTAS = [5, 10, 25, 50, 100]  # |cents| levels probed (psychometric curve)
N_PERM = 200                   # label-permutation null draws
N_SPLITS = 5
RNG = np.random.default_rng(20260725)


# --------------------------------------------------------------------- I/O
def _facts(row):
    return row.factors if isinstance(row.factors, dict) else json.loads(row.factors)


def _npz_path(acts_dir: Path, stimulus_id: str) -> Path:
    return acts_dir / (stimulus_id.replace("/", "__") + ".npz")


def _frame_layers(z) -> list[int]:
    """Layer indices for which frame-level (time x dim) activations were saved."""
    out = []
    for k in z.files:
        if k.startswith("layer_"):
            out.append(int(k.split("_")[1]))
    return sorted(out)


def _infer_fps(n_frames: int) -> float:
    """Frame rate. Whisper/AF3 mel pads to 30 s; others emit the 2.5 s clip.
    Pick the padding interpretation that yields a plausible audio frame rate."""
    for pad in (CLIP_SECONDS, 30.0):
        fps = n_frames / pad
        if 20 <= fps <= 130:
            return fps
    # fall back to the closer-to-50fps interpretation
    cands = [(n_frames / p, p) for p in (CLIP_SECONDS, 30.0)]
    return min(cands, key=lambda c: abs(c[0] - 50))[0]


def _seg_pool(frames: np.ndarray, lo_s: float, hi_s: float, fps: float) -> np.ndarray:
    a, b = int(round(lo_s * fps)), int(round(hi_s * fps))
    a, b = max(0, a), min(frames.shape[0], b)
    if b <= a:
        return frames.mean(axis=0).astype(np.float32)
    return frames[a:b].mean(axis=0).astype(np.float32)


# --------------------------------------------------- significance utilities
def _perm_null_bacc(X, y, groups, n_perm=N_PERM):
    """Empirical p and null distribution for balanced accuracy under label
    permutation (labels shuffled WITHIN the CV, so the null respects group folds)."""
    obs = _cv_bacc(X, y, groups)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = RNG.permutation(y)
        null[i] = _cv_bacc(X, yp, groups)
    p = float((1 + np.sum(null >= obs)) / (n_perm + 1))
    return obs, float(null.mean()), float(null.std()), p


def _cv_bacc(X, y, groups) -> float:
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    if n_splits < 2:
        return np.nan
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)) if accs else np.nan


def _cv_bacc_by_delta(X, y, groups, deltas):
    """Balanced acc overall AND per-|delta| (the psychometric curve), one GroupKFold."""
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    per_delta = {d: [] for d in DELTAS}
    overall = []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        overall.append(balanced_accuracy_score(y[te], pred))
        for d in DELTAS:
            m = deltas[te] == d
            if m.sum() >= 2:
                per_delta[d].append((pred[m] == y[te][m]).mean())
    curve = {d: (float(np.mean(v)) if v else np.nan) for d, v in per_delta.items()}
    return (float(np.mean(overall)) if overall else np.nan), curve


# --------------------------------------------------------------- cents task
def probe_cents(acts_dir: Path, man: pd.DataFrame, encoder: str):
    sub = man[man.task == "cents_discrimination"].reset_index(drop=True)
    sub = sub[sub.ground_truth != "same"].reset_index(drop=True)  # sign needs a direction
    # discover the frame layers this encoder saved
    z0 = np.load(_npz_path(acts_dir, sub.iloc[0].stimulus_id))
    layers = _frame_layers(z0)
    z0.close()
    if not layers:
        print(f"  [{encoder}] no frame-level layers saved -> cannot segment tone pair")
        return []

    # signal-floor oracle: sign of (f2 - f1) from the KNOWN frequencies (L1 check)
    f1 = sub.factors.apply(lambda d: _facts_d(d)["f1_hz"]).to_numpy(float)
    f2 = sub.factors.apply(lambda d: _facts_d(d)["f2_hz"]).to_numpy(float)
    oracle = float((np.sign(f2 - f1) == np.where(sub.ground_truth == "higher", 1, -1)).mean())

    y = (sub.ground_truth == "higher").astype(int).to_numpy()
    delta = sub.factors.apply(lambda d: _facts_d(d)["delta_cents"]).to_numpy(int)
    base = sub.factors.apply(lambda d: _facts_d(d)["base_midi"]).to_numpy(float)
    groups = np.floor(base).astype(int)   # hold out whole base-pitch bins

    rows = []
    for li in layers:
        key = f"layer_{li:02d}"
        D = []
        for r in sub.itertuples():
            z = np.load(_npz_path(acts_dir, r.stimulus_id))
            fr = z[key]
            fps = _infer_fps(fr.shape[0])
            d = _seg_pool(fr, *TONE2, fps) - _seg_pool(fr, *TONE1, fps)
            D.append(d)
            z.close()
        D = np.asarray(D, np.float32)
        obs, nmean, nstd, p = _perm_null_bacc(D, y, groups)
        overall, curve = _cv_bacc_by_delta(D, y, groups, delta)
        rows.append({"encoder": encoder, "task": "cents", "layer": li,
                     "bal_acc": round(obs, 4), "null_mean": round(nmean, 4),
                     "null_std": round(nstd, 4), "p_perm": round(p, 4),
                     "oracle_f0": round(oracle, 4), "n": len(y),
                     **{f"acc_{d}c": (round(curve[d], 4) if not np.isnan(curve[d]) else None)
                        for d in DELTAS}})
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  [{encoder}] cents L{li:02d}: bal_acc={obs:.3f} "
              f"(null {nmean:.3f}±{nstd:.3f}, p={p:.3f} {star})  "
              f"curve " + " ".join(f"{d}c:{curve[d]:.2f}" for d in DELTAS
                                   if not np.isnan(curve[d])))
    print(f"  [{encoder}] signal-floor oracle (sign of f2-f1) = {oracle:.3f}")
    return rows


def _facts_d(d):
    return d if isinstance(d, dict) else json.loads(d)


# -------------------------------------------------------------- tuning task
def probe_tuning(acts_dir: Path, man: pd.DataFrame, encoder: str):
    sub = man[man.task == "tuning_judgment"].reset_index(drop=True)
    y = sub.factors.apply(lambda d: _facts_d(d)["detune_cents"] * _facts_d(d)["sign"]).to_numpy(float)
    base = sub.factors.apply(lambda d: _facts_d(d)["base_midi"]).to_numpy(int)
    groups = base
    snap_mae = float(np.abs(y).mean())   # 12-TET-snap predicts 0 detune always

    # use ALL saved pooled layers (single tone -> clip-pooling is correct here)
    z0 = np.load(_npz_path(acts_dir, sub.iloc[0].stimulus_id))
    pooled_layers = sorted(int(k.split("_")[1]) for k in z0.files if k.startswith("pooled_"))
    z0.close()

    rows = []
    for li in pooled_layers:
        key = f"pooled_{li:02d}"
        X = []
        for r in sub.itertuples():
            z = np.load(_npz_path(acts_dir, r.stimulus_id))
            X.append(z[key].astype(np.float32))
            z.close()
        X = np.asarray(X, np.float32)
        gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
        maes, r2s = [], []
        for tr, te in gkf.split(X, y, groups):
            reg = make_pipeline(StandardScaler(),
                                RidgeCV(alphas=np.logspace(-1, 4, 12)))
            reg.fit(X[tr], y[tr])
            pred = reg.predict(X[te])
            maes.append(np.abs(pred - y[te]).mean())
            ss_res = ((y[te] - pred) ** 2).sum()
            ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
        mae, r2 = float(np.mean(maes)), float(np.nanmean(r2s))
        beats = snap_mae - mae     # positive => encoder recovers real detune
        rows.append({"encoder": encoder, "task": "tuning", "layer": li,
                     "mae_cents": round(mae, 2), "snap_mae_cents": round(snap_mae, 2),
                     "beats_snap_cents": round(beats, 2), "r2": round(r2, 4),
                     "n": len(y)})
        flag = "BEATS grid" if beats > 3 else "~= grid (quantized)"
        print(f"  [{encoder}] tuning L{li:02d}: MAE={mae:5.1f}c  "
              f"snap={snap_mae:.1f}c  gain={beats:+.1f}c  R2={r2:+.3f}  [{flag}]")
    return rows


# --------------------------------------------------------------------- main
def main(acts_root, encoders, manifest, out_dir):
    man = pd.read_parquet(manifest)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    all_cents, all_tuning = [], []
    for enc in encoders:
        acts_dir = Path(acts_root) / enc
        if not acts_dir.exists():
            print(f"[skip] {acts_dir} missing"); continue
        print(f"\n=== {enc} ===")
        all_cents += probe_cents(acts_dir, man, enc)
        all_tuning += probe_tuning(acts_dir, man, enc)
    if all_cents:
        pd.DataFrame(all_cents).to_csv(out / "probe_microtone__cents.csv", index=False)
        print(f"\nwrote {out/'probe_microtone__cents.csv'}")
    if all_tuning:
        pd.DataFrame(all_tuning).to_csv(out / "probe_microtone__tuning.csv", index=False)
        print(f"wrote {out/'probe_microtone__tuning.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="acts")
    ap.add_argument("--encoders", nargs="+",
                    default=["whisper", "mert330", "clap", "qwen25omni_own",
                             "qwen3omni_own", "af3_own", "musicflamingo_own"])
    ap.add_argument("--manifest", default="manifests/stimuli.parquet")
    ap.add_argument("--out", default="results/trackB/probes")
    args = ap.parse_args()
    main(args.acts, args.encoders, args.manifest, args.out)
