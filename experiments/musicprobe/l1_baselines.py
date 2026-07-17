"""L1 baselines — classical DSP answers to the same questions we ask models.

Purpose (plan §0.5): the L1 floor proves the information is physically
recoverable from the rendered audio. If L1 succeeds and a model fails, the
failure is in the model, not the stimulus. Pure numpy/scipy implementations —
good enough for clean synthetic stimuli. (Real-music L1 should use
crepe/librosa/madmom on Colab; this module is the synthetic-battery floor.)

  .venv/bin/python -m musicprobe.l1_baselines
"""
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import find_peaks

from .config import RESULTS_DIR
from .manifest import load_manifest, audio_abspath
from .theory import NOTE_NAMES, MODES


def f0_autocorr(y: np.ndarray, sr: int, fmin=60, fmax=1500) -> float | None:
    """Fundamental via autocorrelation peak with parabolic interpolation
    (sub-sample lag -> cent-level precision on clean tones)."""
    seg = y[len(y) // 4: len(y) // 4 + sr]  # 1 s from the sustained part
    if len(seg) < sr // 4 or np.max(np.abs(seg)) < 1e-4:
        return None
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    lo, hi = int(sr / fmax), int(sr / fmin)
    if hi >= len(ac):
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    # refuse octave-down errors: if half the lag is nearly as strong, take it
    if lag // 2 >= lo and ac[lag // 2] > 0.9 * ac[lag]:
        lag //= 2
    if 1 <= lag < len(ac) - 1:  # parabolic refinement around the peak
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            lag = lag + 0.5 * (a - c) / denom
    return sr / lag


def f0_to_midi(f0: float) -> float:
    return 69 + 12 * np.log2(f0 / 440)


def onset_envelope(y: np.ndarray, sr: int, hop=512):
    frames = len(y) // hop
    e = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2))
                  for i in range(frames)])
    d = np.diff(e, prepend=e[0])
    return np.maximum(d, 0), sr / hop


def tempo_estimate(y: np.ndarray, sr: int) -> float | None:
    env, fps = onset_envelope(y, sr)
    env = env - env.mean()
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    lo, hi = int(fps * 60 / 200), int(fps * 60 / 40)  # 40..200 BPM
    if hi >= len(ac):
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    return 60 * fps / lag


KRUMHANSL_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """Peak-based chroma: only spectral peaks contribute, log-weighted, so
    strong fundamentals don't drown under their own harmonic series."""
    n = 8192
    c = np.zeros(12)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    for start in range(0, len(y) - n, n // 2):
        spec = np.abs(np.fft.rfft(y[start:start + n] * np.hanning(n)))
        idx, _ = find_peaks(spec, height=spec.max() * 0.02)
        for i in idx:
            if 55 < freqs[i] < 1200:
                pc = int(round(69 + 12 * np.log2(freqs[i] / 440))) % 12
                c[pc] += np.log1p(spec[i])
    return c / (c.sum() + 1e-9)


def key_estimate(y: np.ndarray, sr: int) -> str:
    c = chroma(y, sr)
    best, best_r = None, -2
    for tonic in range(12):
        for mode, profile in [("major", KRUMHANSL_MAJ), ("minor", KRUMHANSL_MIN)]:
            r = np.corrcoef(np.roll(profile, tonic), c)[0, 1]
            if r > best_r:
                best_r, best = r, f"{NOTE_NAMES[tonic]} {mode}"
    return best


def run() -> pd.DataFrame:
    man = load_manifest(["pitch_note_id", "cents_discrimination",
                         "tempo_bpm", "key_id"])
    rows = []
    for r in man.itertuples():
        y, sr = sf.read(audio_abspath(r.audio_path))
        task, est, correct = r.task, None, None
        if task == "pitch_note_id":
            f0 = f0_autocorr(y, sr)
            if f0:
                est = NOTE_NAMES[int(round(f0_to_midi(f0))) % 12]
                correct = est == r.ground_truth
        elif task == "cents_discrimination":
            # tone1 = first second, tone2 = last second (0.5 s gap between)
            fa = f0_autocorr(y[:sr], sr)
            fb = f0_autocorr(y[-sr:], sr)
            if fa and fb:
                dc = 1200 * np.log2(fb / fa)
                est = "same" if abs(dc) < 2.5 else ("higher" if dc > 0 else "lower")
                correct = est == r.ground_truth
        elif task == "tempo_bpm":
            bpm = tempo_estimate(y, sr)
            if bpm:
                true = float(r.ground_truth)
                # accept metrical-level (octave) matches as recovered
                ratio = bpm / true
                correct = any(abs(np.log2(ratio) - k) < 0.06 for k in (-1, 0, 1))
                est = round(bpm, 1)
        elif task == "key_id":
            est = key_estimate(y, sr)
            correct = est == r.ground_truth
        rows.append({"stimulus_id": r.stimulus_id, "task": task,
                     "l1_estimate": str(est), "l1_correct": correct})
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS_DIR / "l1_baseline.parquet", index=False)
    print(df.groupby("task")["l1_correct"].agg(["mean", "count"]))
    return df


if __name__ == "__main__":
    run()
