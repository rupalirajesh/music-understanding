"""PitchBench + Track D-zoom front-end (2026-08-21) -- the actual "does the
zoomed-F0-image fix generalize to a genuinely external benchmark" test.

Why this exists: eval_pitchbench.py's --lora-checkpoint flag wraps the D-zoom
checkpoint but only ever sends a plain audio+text prompt -- D-zoom's whole
mechanism depends on the zoomed F0-contour image it was fine-tuned to expect,
so a plain-prompt test only answers "does the checkpoint still behave
reasonably without its expected input," not "does the fix generalize to
PitchBench." This script renders that same image for PitchBench's own audio
(reusing scripts/render_f0_contours.render_zoom unmodified, same pattern as
musicprobe/real_music_nsynth.py's --dzoom-jobs) and feeds BOTH audio + image,
matching what the checkpoint was actually trained on.

Only category A (pitchbench_a1_single_pitch_id) is schema-verified -- same
caveat as eval_pitchbench.py.

  python gpu/eval_pitchbench_dzoom.py --limit 5 --no-lora   # baseline smoke test
  python gpu/eval_pitchbench_dzoom.py --limit 5             # fine-tuned smoke test
  python gpu/eval_pitchbench_dzoom.py --no-lora             # full baseline run
  python gpu/eval_pitchbench_dzoom.py                       # full fine-tuned run
"""
import argparse
import io
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import torch

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
sys.path.insert(0, str(GPU_DIR.parent / "scripts"))
from musicprobe.config import RESULTS_ROOT  # noqa: E402
from train_track_d import load_qwen_omni_for_training  # noqa: E402

HF_DATASET = "pitchbench-authors/PitchBench"
CKPT_DIR = GPU_DIR / "track_d_force_ckpt"
TARGET_SR = 16000


def tag(no_lora=False):
    return f"qwen25omni-{'base' if no_lora else 'zoom'}-pitchbench_dzoom"


def load_model(no_lora=False, seed=0):
    model, processor, _ = load_qwen_omni_for_training()
    if not no_lora:
        from peft import PeftModel
        ckpt_path = CKPT_DIR / f"qwen25omni-zoom-s{seed}"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"{ckpt_path} not found -- train it first: "
                f"python gpu/train_track_d_force.py --seed {seed} --image-kind f0zoom")
        model.thinker = PeftModel.from_pretrained(model.thinker, str(ckpt_path))
    model.eval()
    return model, processor


def generate(processor, model, prompt, wav_path, image_path):
    # Matches train_track_d_force.py's verified pattern exactly: apply_chat_template
    # with tokenize=True/return_dict=True resolves both the audio and image file
    # paths internally -- no separate processor(audio=..., images=...) call needed.
    content = [
        {"type": "audio", "audio": str(wav_path)},
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt},
    ]
    conversation = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(conversation, add_generation_prompt=True,
                                           tokenize=True, return_dict=True,
                                           return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False, return_audio=False)
    if isinstance(out, tuple):
        out = out[0]
    out = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0]


def parse_midi(response: str):
    m = re.search(r"-?\d+", response)
    return int(m.group()) if m else None


def run(out_path: Path, limit: int | None, no_lora: bool, seed: int, config: str):
    import soundfile as sf
    from datasets import load_dataset, Audio
    from render_f0_contours import render_zoom  # noqa: E402

    ds = load_dataset(HF_DATASET, config, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))  # avoid torchcodec, see eval_pitchbench.py
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"{config}: {len(ds)} rows")

    model, processor = load_model(no_lora=no_lora, seed=seed)
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for i, row in enumerate(ds, 1):
            audio_raw = row["audio"]
            if audio_raw.get("bytes") is not None:
                arr, sr = sf.read(io.BytesIO(audio_raw["bytes"]))
            else:
                arr, sr = sf.read(audio_raw["path"])
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            wav_path = tmp / f"clip_{i}.wav"
            sf.write(wav_path, arr, sr)

            img_path = tmp / f"clip_{i}.png"
            # "pitchbench_a1_single_pitch_id" isn't in ZOOM_SPAN -- falls back to
            # ZOOM_DEFAULT_SPAN (450 cents), a wide-enough window for a single
            # absolute-pitch judgement, same default D-zoom uses for tasks it
            # wasn't specifically tuned for (e.g. octave_id).
            render_zoom(wav_path, img_path, config)

            prompt = row["prompt_midi"]
            gt = row["gt_midi"]
            t0 = time.time()
            response = generate(processor, model, prompt, wav_path, img_path)
            elapsed = time.time() - t0
            pred = parse_midi(response)
            results.append({
                "config": config, "prompt": prompt, "response": response,
                "gt_midi": gt, "pred_midi": pred,
                "correct": (pred == gt) if pred is not None else False,
                "source": row.get("source"), "elapsed_s": round(elapsed, 2),
            })
            if i % 10 == 0:
                print(f"  {i}/{len(ds)}")

    n_correct = sum(r["correct"] for r in results)
    print(f"acc={n_correct}/{len(results)} = {n_correct/len(results):.3f}" if results else "(empty)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pitchbench_a1_single_pitch_id")
    ap.add_argument("--limit", type=int, default=None, help="rows, for a smoke test")
    ap.add_argument("--no-lora", action="store_true", help="baseline: base model, no D-zoom adapter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or (RESULTS_ROOT / "external_benchmarks" / f"{tag(a.no_lora)}_{a.config}.json")
    run(out, a.limit, a.no_lora, a.seed, a.config)
