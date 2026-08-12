"""PROJECT_STATE.md next action 24 -- does the D-zoom/E fix (validated on
`cents_discrimination`/`tuning_judgment`, PAPER.md: cents 0.55->0.94, tuning
0.53->0.89) hold up when the base tone is a REAL recorded instrument instead
of a synthesized one?

Ground-truth reality check first: neither task can be answered from an
unmodified real recording at all -- "is this note 37 cents flat" has no
naturally-occurring ground truth outside a controlled synthesis. That's WHY
the original battery generated these two tasks from scratch (numpy/DDSP)
instead of using recordings for anything. The honest way to test the FRONT-
END (not the whole pipeline) against real timbre is a hybrid: start from a
real recorded note (NSynth, real acoustic/electronic instruments, exact MIDI
pitch labels), then apply the SAME KIND of exact, controlled digital pitch-
shift the synthetic stimuli always had -- real timbre, still-exact ground
truth. This module builds that, and says so plainly in every stimulus's
provenance rather than presenting it as "real recordings, no synthesis
involved" -- it isn't; only the base timbre is real.

Source: NSynth ('pitch' config, HuggingFace `confit/nsynth-parquet`) --
confirmed live 2026-08-12 (PROJECT_STATE next action 23's dataset hunt).
Pulled via `datasets.Audio(decode=False)` + manual soundfile decode, NOT
`torchcodec` -- avoids pulling torch onto this laptop, which every other
module in this project deliberately keeps GPU-only.

  python -m musicprobe.real_music_nsynth --n 60 --seed 0
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from .config import EXP_ROOT, MANIFEST_DIR
from .theory import NOTE_NAMES
from .l1_baselines import f0_autocorr, f0_to_midi, tuning_estimate
from .prompts import build_prompt

OUT_AUDIO_DIR = Path("stimuli") / "real_nsynth"
MANIFEST_PATH = MANIFEST_DIR / "real_nsynth_manifest.parquet"
JOBS_PATH = MANIFEST_DIR / "real_nsynth_jobs.parquet"

TARGET_SR = 16000            # NSynth's native rate
NOTE_DUR = 1.0                # seconds of clean sustain kept per tone (trimmed
                              # from NSynth's 4s note+decay -- long enough for
                              # f0_autocorr's own 1s analysis window)
GAP_DUR = 0.3                 # silence between tone1/tone2 in cents_discrimination
                              # clips, mirrors the synthetic battery's pause
CENTS_DELTAS = [5, 10, 25, 50, 100]     # same psychometric ladder as TASKS.md 1.8
TUNING_THRESHOLD = 25.0       # matches l1_baselines.tuning_estimate exactly


def _fetch_pool(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset, Audio
    ds = load_dataset("confit/nsynth-parquet", "pitch", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    rng = np.random.default_rng(seed)
    pool = []
    # reservoir-sample-ish: pull a generous prefix, then randomly subsample,
    # since this is a streaming split with no direct random-access indexing
    prefix_n = max(n * 15, 400)
    seen = []
    for i, ex in enumerate(ds):
        seen.append(ex)
        if i + 1 >= prefix_n:
            break
    idx = rng.choice(len(seen), size=min(n, len(seen)), replace=False)
    for i in idx:
        ex = seen[i]
        y, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
        if y.ndim > 1:
            y = y.mean(axis=1)
        pool.append({"y": y.astype(np.float32), "sr": sr, "midi": int(ex["pitch"]),
                    "path": ex["audio"]["path"]})
    return pool


def _sustain(note: dict) -> np.ndarray:
    """First NOTE_DUR seconds after a short onset skip -- avoids the attack
    transient and stays well clear of NSynth's decay tail."""
    y, sr = note["y"], note["sr"]
    onset_skip = int(0.05 * sr)
    n = int(NOTE_DUR * sr)
    seg = y[onset_skip:onset_skip + n]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    return seg


def _shift(seg: np.ndarray, sr: int, cents: float) -> np.ndarray:
    if cents == 0:
        return seg
    return librosa.effects.pitch_shift(seg, sr=sr, n_steps=cents / 100.0)


def build_pool_and_stimuli(n: int, seed: int, exp_root: Path = EXP_ROOT) -> pd.DataFrame:
    (exp_root / OUT_AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    pool = _fetch_pool(n, seed)
    rng = np.random.default_rng(seed)
    rows = []

    # ---- pitch_note_id: one clean sustained real note per pool item ----
    for note in pool:
        seg = _sustain(note)
        stim_id = f"real_nsynth/pitch/{Path(note['path']).stem}"
        rel = str(OUT_AUDIO_DIR / f"pitch_{Path(note['path']).stem}.wav")
        sf.write(exp_root / rel, seg, note["sr"])
        note_name = NOTE_NAMES[note["midi"] % 12]
        rows.append({"stimulus_id": stim_id, "task": "pitch_note_id", "audio_path": rel,
                    "ground_truth": note_name, "source": "nsynth_real",
                    "provenance": "real recording, unmodified", "base_file": note["path"]})

    # ---- cents_discrimination: real tone1 + (shifted or not) real tone2 ----
    labels = ["same", "higher", "lower"]
    for i, note in enumerate(pool):
        seg = _sustain(note)
        label = labels[i % 3]
        if label == "same":
            cents = 0.0
        else:
            delta = CENTS_DELTAS[i % len(CENTS_DELTAS)]
            cents = delta if label == "higher" else -delta
        tone2 = _shift(seg, note["sr"], cents)
        gap = np.zeros(int(GAP_DUR * note["sr"]), dtype=np.float32)
        clip = np.concatenate([seg, gap, tone2])
        stim_id = f"real_nsynth/cents/{Path(note['path']).stem}_{label}_{abs(cents):.0f}c"
        rel = str(OUT_AUDIO_DIR / f"cents_{Path(note['path']).stem}_{label}_{abs(cents):.0f}c.wav")
        sf.write(exp_root / rel, clip, note["sr"])
        rows.append({"stimulus_id": stim_id, "task": "cents_discrimination", "audio_path": rel,
                    "ground_truth": label, "source": "nsynth_real+controlled_shift",
                    "provenance": f"real recording base, tone2 digitally shifted {cents:+.0f} "
                                  "cents (ground truth requires an exact known shift -- not "
                                  "obtainable from an unmodified recording, see module docstring)",
                    "base_file": note["path"], "cents_shift": cents})

    # ---- tuning_judgment: real tone, in tune or controlled-shifted out of tune ----
    for i, note in enumerate(pool):
        seg = _sustain(note)
        out_of_tune = (i % 2 == 1)
        if out_of_tune:
            cents = float(rng.uniform(TUNING_THRESHOLD + 5, 95))  # clear of the 25c threshold
            cents *= rng.choice([-1, 1])
            clip = _shift(seg, note["sr"], cents)
            label = "out of tune"
        else:
            clip, cents, label = seg, 0.0, "in tune"
        stim_id = f"real_nsynth/tuning/{Path(note['path']).stem}_{label.replace(' ', '')}"
        rel = str(OUT_AUDIO_DIR / f"tuning_{Path(note['path']).stem}_{label.replace(' ', '')}.wav")
        sf.write(exp_root / rel, clip, note["sr"])
        rows.append({"stimulus_id": stim_id, "task": "tuning_judgment", "audio_path": rel,
                    "ground_truth": label, "source": "nsynth_real+controlled_shift",
                    "provenance": ("real recording, unmodified" if cents == 0 else
                                  f"real recording base, digitally shifted {cents:+.0f} cents "
                                  "(ground truth requires an exact known shift, see module "
                                  "docstring)"),
                    "base_file": note["path"], "cents_shift": cents})

    manifest = pd.DataFrame(rows)
    # `factors` column (PROJECT_STATE next action 25, added 2026-08-12): held-out-group
    # key for gpu/probe.py's leakage-guard folds, same role as the synthetic battery's
    # factors.soundfont -- grouped by INSTRUMENT FAMILY (NSynth's own path convention,
    # e.g. "organ_electronic", parsed as everything before the first "<3-digit>-<pitch>-
    # <velocity>" segment) so a probe never sees the same instrument in both train and
    # held folds -- guards against timbre-fingerprint leakage exactly like soundfont does.
    import json as _json
    import re as _re

    def _family(base_file):
        m = _re.match(r"^(.*?)_\d+-\d+-\d+", base_file)
        return m.group(1) if m else base_file
    manifest["factors"] = manifest["base_file"].apply(
        lambda f: _json.dumps({"instrument_family": _family(f)}))
    return manifest


def build_jobs(manifest: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    job_rows = []
    for i, r in enumerate(manifest.itertuples()):
        others = manifest[(manifest.task == r.task) & (manifest.stimulus_id != r.stimulus_id)]
        wrong = others.sample(1, random_state=seed + i).iloc[0] if len(others) else None
        for pi in range(3):
            prompt = build_prompt(r.task, pi, "open", None)
            job_rows.append({"job_id": f"{r.stimulus_id}::audio::open::p{pi}",
                            "stimulus_id": r.stimulus_id, "task": r.task, "condition": "audio",
                            "format": "open", "paraphrase_idx": pi, "prompt": prompt,
                            "ground_truth": r.ground_truth, "audio_path": r.audio_path})
            if wrong is not None:
                job_rows.append({"job_id": f"{r.stimulus_id}::wrong_audio::open::p{pi}",
                                "stimulus_id": r.stimulus_id, "task": r.task,
                                "condition": "wrong_audio", "format": "open",
                                "paraphrase_idx": pi, "prompt": prompt,
                                "ground_truth": r.ground_truth, "audio_path": wrong.audio_path})
    return pd.DataFrame(job_rows)


def build_manifest(n: int = 60, seed: int = 0, exp_root: Path = EXP_ROOT):
    man = build_pool_and_stimuli(n, seed, exp_root)
    jobs = build_jobs(man, seed)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    man.to_parquet(MANIFEST_PATH, index=False)
    jobs.to_parquet(JOBS_PATH, index=False)
    print(f"[real_nsynth] {len(man)} stimuli -> {MANIFEST_PATH} "
          f"({man.task.value_counts().to_dict()})")
    print(f"[real_nsynth] {len(jobs)} jobs -> {JOBS_PATH} "
          f"({jobs.condition.value_counts().to_dict()})")
    return man, jobs


def l1_accuracy(manifest_path: Path = MANIFEST_PATH, exp_root: Path = EXP_ROOT) -> pd.DataFrame:
    """L1's own DSP estimators (l1_baselines.f0_autocorr/tuning_estimate,
    UNCHANGED) run against these hybrid real+shifted stimuli -- the same
    "does the pipeline's own algorithm work on this audio at all" sanity
    check as the Bach/Debussy/real-recordings passes (PROJECT_STATE next
    action 23), not a model result. cents_discrimination reuses the EXACT
    tone1=first-1s/tone2=last-1s slicing l1_baselines.run() already assumes,
    which is why NOTE_DUR/GAP_DUR above are sized so that slicing lands
    exactly on tone1/tone2."""
    man = pd.read_parquet(manifest_path)
    rows = []
    for r in man.itertuples():
        y, sr = sf.read(exp_root / r.audio_path)
        def _valid(f0):   # f0_autocorr can return a non-positive/NaN lag-derived
                          # value on real (non-clean-synthetic) audio at edge cases --
                          # not something the earlier synthetic-only L1 battery hit
            return f0 is not None and np.isfinite(f0) and f0 > 0

        if r.task == "tuning_judgment":
            try:
                est = tuning_estimate(y, sr, cents_threshold=TUNING_THRESHOLD)
            except (ValueError, OverflowError):
                # l1_baselines.tuning_estimate is existing, already-verified code
                # (used for the synthetic battery) -- not modifying it here, just
                # guarding against the NaN/negative-f0 edge cases real audio can
                # trigger that the synthetic-only battery never exercised
                est = None
        elif r.task == "cents_discrimination":
            fa = f0_autocorr(y[:sr], sr)
            fb = f0_autocorr(y[-sr:], sr)
            if _valid(fa) and _valid(fb):
                dc = 1200 * np.log2(fb / fa)
                est = "same" if abs(dc) < 2.5 else ("higher" if dc > 0 else "lower")
            else:
                est = None
        elif r.task == "pitch_note_id":
            f0 = f0_autocorr(y, sr)
            est = NOTE_NAMES[int(round(f0_to_midi(f0))) % 12] if _valid(f0) else None
        else:
            continue
        rows.append({"task": r.task, "stimulus_id": r.stimulus_id,
                    "ground_truth": r.ground_truth, "l1_estimate": est,
                    "correct": est == r.ground_truth})
    df = pd.DataFrame(rows)
    summary = df.groupby("task")["correct"].agg(["mean", "count"]).rename(
        columns={"mean": "l1_acc", "count": "n"})
    print(summary.to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="notes per task family")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l1-only", action="store_true",
                    help="skip fetching, just re-run L1 accuracy against an existing manifest")
    a = ap.parse_args()
    if a.l1_only:
        l1_accuracy()
    else:
        build_manifest(a.n, a.seed)
        l1_accuracy()
