"""Large mode_id + key_id set to POWER-TEST the near-floor nulls (2026-08-13).

The battery has only ~104 mode / ~96 key stimuli — at 13/24 classes and a
held-out-SOUNDFONT split that's ~5/3 examples per class per fold, too thin to
certify "mode is genuinely absent" vs "the probe lacked power." This regenerates
the SAME stimulus construction (reuses key_mode.py's _scale_notes/_progression_
notes unchanged) at ~10x scale, spread explicitly across all 3 soundfonts so the
leakage guard still works. Distinct IDs (augm_/augk_), separate manifest — never
touches the frozen battery.

  python scripts/generate_aug_keymode.py
"""
import argparse
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from musicprobe.config import STIMULI_DIR, EXP_ROOT, GM_PROGRAMS, MANIFEST_DIR, available_soundfonts
from musicprobe.synth import midi_notes, render_midi
from musicprobe.theory import MODES, MODE_SPOKEN, key_name
from musicprobe.generators.key_mode import _scale_notes, _progression_notes

AUG_KM_PATH = MANIFEST_DIR / "aug_keymode_manifest.parquet"
MODE_TONICS = [53, 55, 57, 59, 60, 62, 64, 65]     # F3..F4, 8 tonics
KEY_TONICS = list(range(12))


def _seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def gen_mode(reps, progs):
    sfs = list(available_soundfonts().items())
    rows = []
    for mode in MODES:
        for tonic in MODE_TONICS:
            for form in ["bare_scale", "melody_drone"]:
                for sf_name, sf_path in sfs:
                    for rep in range(reps):
                        r = np.random.default_rng(_seed("m", mode, tonic, form, sf_name, rep))
                        if form == "bare_scale":
                            notes = _scale_notes(tonic, MODES[mode])
                        else:
                            deg = MODES[mode]; mel, t = [], 0.25
                            for _ in range(16):
                                p = tonic + deg[int(r.integers(len(deg)))] + 12 * int(r.integers(0, 2))
                                d = float(r.choice([0.25, 0.5, 0.5, 0.75]))
                                mel.append((p, t, t + d * 0.95)); t += d
                            notes = mel + [(tonic - 12, 0.25, t), (tonic - 24, 0.25, t)]
                        prog_name, prog = progs[r.integers(len(progs))]
                        sid = f"augm_{mode}_{tonic}_{form}_{sf_name}_{rep}"
                        rel = f"stimuli/mode/{sid}.wav"
                        render_midi(midi_notes(notes, program=prog), sf_path, EXP_ROOT / rel)
                        rows.append({"stimulus_id": f"mode_id/{sid}", "task": "mode_id",
                                     "audio_path": rel, "ground_truth": MODE_SPOKEN[mode],
                                     "factors": {"mode": mode, "tonic_midi": tonic, "form": form,
                                                 "program": prog_name, "soundfont": sf_name}})
    return rows


def gen_key(reps, progs):
    sfs = list(available_soundfonts().items())
    from musicprobe.theory import NOTE_NAMES
    rows = []
    for tonic_pc in KEY_TONICS:
        for mode in ["major", "minor"]:
            for form in ["scale", "progression"]:
                for sf_name, sf_path in sfs:
                    for rep in range(reps):
                        r = np.random.default_rng(_seed("k", tonic_pc, mode, form, sf_name, rep))
                        tonic_midi = 60 + tonic_pc - (12 if tonic_pc > 6 else 0)
                        if form == "scale":
                            deg = MODES["major" if mode == "major" else "natural_minor"]
                            notes = _scale_notes(tonic_midi, deg)
                        else:
                            notes = _progression_notes(tonic_midi, mode, r)
                        prog_name, prog = progs[r.integers(len(progs))]
                        sid = f"augk_{NOTE_NAMES[tonic_pc].replace('#','s')}_{mode}_{form}_{sf_name}_{rep}"
                        rel = f"stimuli/key/{sid}.wav"
                        render_midi(midi_notes(notes, program=prog), sf_path, EXP_ROOT / rel)
                        rows.append({"stimulus_id": f"key_id/{sid}", "task": "key_id",
                                     "audio_path": rel, "ground_truth": key_name(tonic_pc, mode),
                                     "factors": {"tonic_pc": tonic_pc, "mode": mode, "form": form,
                                                 "program": prog_name, "soundfont": sf_name}})
    return rows


def main(mode_reps, key_reps):
    progs = list(GM_PROGRAMS.items())
    rows = gen_mode(mode_reps, progs) + gen_key(key_reps, progs)
    df = pd.DataFrame(rows)
    df.to_parquet(AUG_KM_PATH, index=False)
    print(f"[aug_keymode] {len(df)} stimuli "
          f"({(df.task=='mode_id').sum()} mode / {(df.task=='key_id').sum()} key) -> {AUG_KM_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode-reps", type=int, default=2)   # 13x8x2x3x2 = 1248 mode
    ap.add_argument("--key-reps", type=int, default=4)    # 24x2x3x4   = 576 key
    a = ap.parse_args()
    main(a.mode_reps, a.key_reps)
