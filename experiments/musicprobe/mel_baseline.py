"""PROJECT_STATE.md next action 20 -- a third floor alongside L1 (hand-
designed DSP estimators, l1_baselines.py) and L2 (linear probe on a
PRETRAINED encoder's activations, gpu/probe.py). This one asks: with no
hand-designed features and no pretrained encoder at all, how much is
recoverable from the raw mel-spectrogram alone, fed straight into a linear
classifier?

Same held-out-soundfont-fold discipline as gpu/probe.py's probe_layer
(duplicated here, not imported, so this CPU-only/no-torch module has zero
dependency on gpu/ -- consistent with l1_baselines.py's own standalone
design). Unlike L1, which is blocked on essentia for beats_per_bar/
progression_id/instrument_id (musicprobe/l1_baselines.py's TASKS_WITH_L1
comment), this baseline needs no beat/onset-tracking library at all -- it's
just a spectrogram and a classifier -- so it can attempt all 13 tasks.

  python -m musicprobe.mel_baseline
"""
import json

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import EXP_ROOT, MANIFEST_PATH, RESULTS_ROOT

TARGET_SR = 22050
N_MELS = 64
OUT_PATH = RESULTS_ROOT / "mel_baseline.parquet"
HELD_OUT_SOUNDFONTS = {"timgm"}   # same constant as gpu/train_track_c.py, duplicated
                                   # (not imported) to keep this module torch-free

ALL_TASKS = ["pitch_note_id", "cents_discrimination", "tempo_bpm", "key_id", "octave_id",
             "tuning_judgment", "note_count", "interval_id", "chord_quality", "mode_id",
             "beats_per_bar", "progression_id", "instrument_id"]


def _load(wav_path) -> np.ndarray:
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)


def mel_features(wav_path) -> np.ndarray:
    """Fixed-size feature regardless of clip length: log-mel spectrogram,
    mean+std pooled over time per mel bin -> 2*N_MELS-dim vector. No
    hand-picked musical structure (no peak-picking, no onset detection,
    no pitch tracking) -- exactly the raw time-frequency representation
    every model in the roster computes internally (RESEARCH_PLAN.md Sec0.2),
    with nothing musical added on top."""
    y = _load(wav_path)
    if len(y) < 512:
        return np.zeros(2 * N_MELS, dtype=np.float32)
    S = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS, hop_length=512)
    logS = librosa.power_to_db(S, ref=np.max)
    return np.concatenate([logS.mean(axis=1), logS.std(axis=1)]).astype(np.float32)


def _held_out_mask(factors: pd.Series) -> pd.Series:
    """Same 3-tier discipline as gpu/image_track_common._held_out_mask
    (soundfont membership -> held-out top-quintile base_midi -> held-out
    top-quintile bpm), duplicated here rather than imported so this module
    stays torch-free (gpu/train_track_c.py, which defines the base tier,
    imports torch at module level). A single train/held split, not K-fold --
    matches every LoRA track's discipline in this project rather than
    inventing a separate CV scheme for this one baseline. Caught by testing
    2026-08-12: beats_per_bar has NEITHER soundfont NOR base_midi, so without
    this third tier the group column collapses to one bucket and the
    fallback becomes an unprincipled random split -- exactly the same bug
    class as PROJECT_STATE.md next action 15a, now guarded against here too."""
    has_sf = factors.apply(lambda d: "soundfont" in d)
    has_midi = factors.apply(lambda d: "base_midi" in d)
    mask = pd.Series(False, index=factors.index)
    if has_sf.any():
        mask[has_sf] = factors[has_sf].apply(lambda d: d["soundfont"] in HELD_OUT_SOUNDFONTS)
    tier2 = has_midi & ~has_sf
    if tier2.any():
        midi = pd.to_numeric(factors[tier2].apply(lambda d: d.get("base_midi")), errors="coerce")
        mask[tier2] = midi >= midi.quantile(0.8)
    tier3 = ~(has_sf | has_midi)
    if tier3.any():
        bpm = pd.to_numeric(factors[tier3].apply(lambda d: d.get("bpm")), errors="coerce")
        if bpm.notna().any():
            mask[tier3] = bpm >= bpm.quantile(0.8)
        else:
            # last resort, no known leakage-relevant factor at all (e.g.
            # progression_id): seeded random 80/20, flagged in the output.
            idx = np.random.default_rng(0).permutation(tier3.sum())
            cut = int(0.8 * tier3.sum())
            rmask = np.zeros(tier3.sum(), dtype=bool); rmask[idx[cut:]] = True
            mask[tier3] = rmask
    return mask


def run(tasks=ALL_TASKS, seed=0) -> pd.DataFrame:
    man = pd.read_parquet(MANIFEST_PATH)
    rows = []
    for task in tasks:
        sub = man[man.task == task].reset_index(drop=True)
        if len(sub) == 0:
            print(f"  {task}: 0 stimuli in manifest, skipping"); continue
        factors = sub.factors.apply(lambda f: f if isinstance(f, dict) else json.loads(f))
        split_tier = ("soundfont" if factors.apply(lambda d: "soundfont" in d).any()
                      else "base_midi" if factors.apply(lambda d: "base_midi" in d).any()
                      else "bpm" if factors.apply(lambda d: "bpm" in d).any()
                      else "random-no-factor")
        ho = _held_out_mask(factors).values
        y_all = sub.ground_truth.values
        if len(np.unique(y_all)) < 2 or ho.sum() < 5 or (~ho).sum() < 5:
            print(f"  {task}: degenerate split (held={ho.sum()}, train={(~ho).sum()}, "
                  f"classes={len(np.unique(y_all))}), skipping"); continue
        X_all = np.array([mel_features(EXP_ROOT / p) for p in sub.audio_path])
        Xtr, Xte = X_all[~ho], X_all[ho]
        ytr, yte = y_all[~ho], y_all[ho]
        if len(np.unique(ytr)) < 2:
            print(f"  {task}: <2 classes in train split, skipping"); continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Xtr, ytr)
        acc = float(clf.score(Xte, yte))
        chance = 1 / len(np.unique(y_all))
        rows.append({"task": task, "mel_baseline_acc": acc, "n_train": len(Xtr),
                     "n_held": len(Xte), "chance": chance, "split_tier": split_tier})
        print(f"  {task:20s} train={len(Xtr):4d} held={len(Xte):4d} acc={acc:.3f} "
              f"(chance {chance:.3f}, split={split_tier})")
    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"[mel_baseline] wrote {OUT_PATH}")
    return out


if __name__ == "__main__":
    run()
