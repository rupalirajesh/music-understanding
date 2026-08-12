"""Auxiliary self-transcription training TARGET (RESEARCH_PLAN.md Sec 12.3,
PROJECT_STATE.md next action 17) -- resolves the format blocker that section
was parked on.

Format decision (this file): a compact JSON event list of audio-derived
(onset, duration, pitch-in-Hz) triples, e.g.
  [{"onset": 0.00, "dur": 0.42, "hz": 261.6}, {"onset": 0.42, "dur": 0.38, "hz": 329.6}]
Chosen over plain MIDI-as-text because pitch is continuous Hz, not a
quantized note number -- satisfies RUPALI_READ_THIS.md Sec5's "no continuous
pitch" objection to MIDI while staying compact and human-readable. Detection
logic mirrors render_harmony_repr.render_piano_roll's onset+piptrack chain
exactly (same non-leakage discipline: audio-derived via librosa, never from
MIDI/factors ground truth) but is DUPLICATED here rather than imported, so
this new consumer can never change Track P's already-landed behavior.

This is a TRAINING TARGET only (Phase 2's auxiliary objective is trained to
predict this string, then the output head is discarded at test time) -- it is
never shown to the model as an input, so unlike every image/text FRONT-END
track in this project, leakage-into-the-prompt is not the risk here; the risk
this file guards against is accidentally forking Track P's detection code and
having the two silently drift.

  python -m musicprobe.transcription_target        # build + cache the map
"""
import json

import numpy as np
import soundfile as sf
import librosa
import pandas as pd

from .config import EXP_ROOT, MANIFEST_DIR, MANIFEST_PATH

TRANSCRIPTION_TARGET_PATH = MANIFEST_DIR / "transcription_target.json"
TARGET_SR = 22050
HOP = 512
FMIN, FMAX = 80.0, 1200.0
MAX_NOTES_PER_SEGMENT = 4   # cap, same rationale as render_piano_roll: don't
                            # let one noisy onset-segment flood the target


def _load(wav_path) -> np.ndarray:
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)


def _reject_harmonics(freqs: list[float], tol: float = 0.03) -> list[float]:
    """Greedy fundamental-only filter -- duplicated verbatim from
    render_harmony_repr._reject_harmonics (see module docstring for why)."""
    cand = sorted({f for f in freqs if f > 0})
    funds: list[float] = []
    for f in cand:
        if not any(abs(f / g - round(f / g)) < tol and round(f / g) >= 1 for g in funds):
            funds.append(f)
    return funds


def note_events(wav_path) -> list[dict]:
    """Audio-derived (onset, dur, hz) triples, time order. Same onset-detect +
    per-segment piptrack + harmonic-rejection chain as
    render_harmony_repr.render_piano_roll, minus the plotting."""
    y = _load(wav_path)
    onset_frames = librosa.onset.onset_detect(y=y, sr=TARGET_SR, hop_length=HOP, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=TARGET_SR, hop_length=HOP)
    duration = len(y) / TARGET_SR
    bounds = list(onset_times) + [duration]
    if len(bounds) < 2:
        bounds = [0.0, duration]
    events = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        seg = y[int(t0 * TARGET_SR):int(t1 * TARGET_SR)]
        if len(seg) < 512:
            continue
        pitches, mags = librosa.piptrack(y=seg, sr=TARGET_SR, hop_length=HOP, fmin=FMIN, fmax=FMAX)
        col_mag = mags.max(axis=1)
        if col_mag.max() <= 0:
            continue
        thresh = col_mag.max() * 0.2
        raw = []
        for bin_idx in np.where(col_mag >= thresh)[0]:
            row = mags[bin_idx, :]
            f = pitches[bin_idx, np.argmax(row)]
            if f > 0:
                raw.append(f)
        for f in _reject_harmonics(raw)[:MAX_NOTES_PER_SEGMENT]:
            events.append({"onset": round(float(t0), 3), "dur": round(float(t1 - t0), 3),
                           "hz": round(float(f), 1)})
    return events


def text_for_events(events: list[dict]) -> str:
    if not events:
        return "[]"
    return json.dumps(events, separators=(",", ":"))


def build_map(tasks=None) -> dict:
    """tasks=None -> every stimulus in the manifest (Phase 2 runs across all
    three clusters -- pitch, harmony, rhythm -- per next action 17, unlike
    Track E's f0_text which was pitch-only)."""
    man = pd.read_parquet(MANIFEST_PATH)
    if tasks is not None:
        man = man[man.task.isin(tasks)]
    out = {}
    n_errors = 0
    for i, r in enumerate(man.itertuples(), 1):
        try:
            out[r.audio_path] = text_for_events(note_events(EXP_ROOT / r.audio_path))
        except Exception as e:
            out[r.audio_path] = "[]"
            n_errors += 1
        if i % 100 == 0:
            print(f"  transcription_target {i}/{len(man)}")
    TRANSCRIPTION_TARGET_PATH.write_text(json.dumps(out))
    print(f"[transcription_target] wrote {TRANSCRIPTION_TARGET_PATH} "
          f"({len(out)} stimuli, {n_errors} errors)")
    return out


def load_map() -> dict:
    if TRANSCRIPTION_TARGET_PATH.exists():
        return json.loads(TRANSCRIPTION_TARGET_PATH.read_text())
    return build_map()


if __name__ == "__main__":
    build_map()
