"""Generate the constraint-adherence battery with MusicGen (runs on H100).

  pip install torch transformers accelerate soundfile pandas pyarrow
  python genmodel/run_musicgen.py --model facebook/musicgen-medium --out genmodel/outputs

Design rules (from discussion):
- ONE constraint per prompt, never compounded ("97 BPM in F minor" tells us
  nothing about which constraint failed).
- Constraint values are non-round and varied (97 BPM, not 120) so satisfying
  them by luck is detectable.
- Every constraint family has an UNCONSTRAINED baseline pool: adherence is
  reported as P(satisfied | asked) − P(satisfied | not asked). If unprompted
  output is in 4/4 at ~120 BPM half the time, raw adherence numbers lie.
- Vocabulary-vs-theory pairs (the Suno waltz observation, made controlled):
  the same musical constraint phrased as a style word ("a waltz") vs a theory
  term ("in 3/4 time"). If adherence tracks phrasing rather than the
  constraint itself, the model learned caption vocabulary, not music.
"""
import argparse
from pathlib import Path

import pandas as pd

N_PER_PROMPT = 12      # generations per prompt variant
N_BASELINE = 30        # unconstrained generations (base-rate pool)
DUR_S = 12

BASE = "instrumental music"

PROMPTS = []  # (family, phrasing_kind, prompt, constraint_key, constraint_value)

# G1 tempo — theory phrasing only (no style word implies an exact BPM)
for bpm in [72, 84, 97, 113, 126, 143, 158, 176]:
    PROMPTS.append(("tempo", "theory", f"{BASE} at {bpm} beats per minute", "bpm", bpm))

# G2 meter — the vocabulary-vs-theory pairs
PROMPTS += [
    ("meter", "vocab",  "a waltz",                        "beats_per_bar", 3),
    ("meter", "theory", f"{BASE} in 3/4 time",            "beats_per_bar", 3),
    ("meter", "vocab",  "a military march",               "beats_per_bar", 4),
    ("meter", "theory", f"{BASE} in 4/4 time",            "beats_per_bar", 4),
    ("meter", "theory", f"{BASE} in 5/4 time",            "beats_per_bar", 5),
    ("meter", "theory", f"{BASE} in 7/8 time",            "beats_per_bar", 7),
]

# G3 key/mode — theory phrasing; include a vocab twin where one exists
for key in ["C major", "E major", "F# minor", "B minor", "Eb major", "G minor"]:
    PROMPTS.append(("key", "theory", f"{BASE} in the key of {key}", "key", key))
PROMPTS += [
    ("mode", "theory", f"{BASE} in D Dorian mode",        "mode", "D dorian"),
    ("mode", "vocab",  "a folk tune with a Celtic modal sound",  "mode", None),  # exploratory
    ("mode", "theory", f"{BASE} using only the whole-tone scale", "mode", "whole_tone"),
    ("mode", "vocab",  "dreamy impressionist music like Debussy", "mode", None),  # exploratory
]

# G4 register — single constraint, solo instrument so f0 tracking is meaningful
PROMPTS += [
    ("register", "theory", "a solo flute melody staying above C5 at all times",
     "min_midi", 72),
    ("register", "theory", "a solo bass line staying below C3 at all times",
     "max_midi", 48),
]


def main(model_name: str, out_dir: str):
    import torch
    import soundfile as sf
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32).to("cuda").eval()
    sr = model.config.audio_encoder.sampling_rate
    max_tokens = int(DUR_S * 50)  # EnCodec at 50 Hz

    rows = []
    battery = [("baseline", "none", BASE, None, None)] * N_BASELINE + \
              [p for p in PROMPTS for _ in range(N_PER_PROMPT)]
    for i, (family, phrasing, prompt, ckey, cval) in enumerate(battery):
        fname = out / f"{family}_{phrasing}_{i:04d}.wav"
        if not fname.exists():
            inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
            with torch.no_grad():
                audio = model.generate(**inputs, do_sample=True,
                                       guidance_scale=3.0,
                                       max_new_tokens=max_tokens)
            sf.write(fname, audio[0, 0].cpu().float().numpy(), sr)
        rows.append({"file": fname.name, "family": family, "phrasing": phrasing,
                     "prompt": prompt, "constraint_key": ckey,
                     "constraint_value": str(cval), "model": model_name})
        if i % 20 == 0:
            print(f"{i}/{len(battery)}")
            pd.DataFrame(rows).to_parquet(out / "generations.parquet", index=False)
    pd.DataFrame(rows).to_parquet(out / "generations.parquet", index=False)
    print(f"done: {len(rows)} clips in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/musicgen-medium")
    ap.add_argument("--out", default="genmodel/outputs")
    args = ap.parse_args()
    main(args.model, args.out)
