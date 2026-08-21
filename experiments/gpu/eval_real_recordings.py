"""Base-model real-recordings key-identification eval (2026-08-21).

Asks Qwen2.5-Omni-7B "what key is this piece in?" against 7 real, free-licensed
recordings pulled from Wikimedia Commons (manifests/real_recordings_manifest.csv),
each with a documented, cited key -- not a musician's judgment call, a citable
published fact (see DIRECTION_DECISION.md's "ground truth by citation" scope
note). 4 "famous" pieces (text-recall risk) + 3 "obscure" pieces (lower recall
risk, same citation quality) -- this split lets us see whether the model is
reciting a memorized fact vs actually engaging with the audio.

This is the genuine "real music" test (full performances, not isolated notes
like NSynth) -- base model only, no fine-tuning involved. Logs the full raw
response verbatim for every item, per the project's "log everything" rule --
nothing is scored away, only a loose substring hint is added for a quick read.

  python gpu/eval_real_recordings.py
  python gpu/eval_real_recordings.py --limit 2   # smoke test first
"""
import argparse
import json
import time
from pathlib import Path

import torch

GPU_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
from musicprobe.config import EXP_ROOT, RESULTS_ROOT, MANIFEST_DIR  # noqa: E402

MANIFEST_PATH = MANIFEST_DIR / "real_recordings_manifest.csv"
PROMPT = "What musical key is this piece in? Answer with the key (e.g. 'C major' or 'A minor') and a brief reason."


def load_model():
    from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto").eval()
    # NOT calling disable_talker() -- confirmed 2026-08-21: crashes generate()
    # on this transformers version regardless of return_audio. See
    # eval_pitchbench.py's load_model() for the full explanation.
    return processor, model


def generate(processor, model, wav):
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio": "stimulus.wav"},
        {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[wav], sampling_rate=16000,
                        return_tensors="pt", padding=True).to(model.device)
    for k in list(inputs.keys()):
        if torch.is_floating_point(inputs[k]):
            inputs[k] = inputs[k].to(model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False, return_audio=False)
    if isinstance(out, tuple):
        out = out[0]
    out = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0]


def run(out_path: Path, limit: int | None = None):
    import csv
    import librosa

    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    print(f"real_recordings: {len(rows)} rows")

    processor, model = load_model()
    results = []
    for i, row in enumerate(rows, 1):
        audio_path = EXP_ROOT / row["audio_path"]
        # librosa.load handles both .ogg (audioread fallback) and .wav via one
        # call -- avoids re-deriving per-format loading logic for 2 formats.
        wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        t0 = time.time()
        response = generate(processor, model, wav)
        elapsed = time.time() - t0
        documented_key = row["key"]
        # loose hint only, NOT a scored verdict -- open-format key answers can
        # be phrased many valid ways (enharmonics, "C# minor" vs "C-sharp
        # minor", major/minor implied vs stated); this is a fast eyeball aid,
        # real correctness needs a human read of the raw response.
        key_mentioned_hint = documented_key.lower().replace("-", " ") in response.lower().replace("-", " ")
        results.append({
            "audio_path": row["audio_path"], "composer": row["composer"], "title": row["title"],
            "documented_key": documented_key, "key_source": row["key_source"],
            "fame_tier": row["fame_tier"], "source_url": row["source_url"],
            "raw_response": response, "key_mentioned_hint": key_mentioned_hint,
            "elapsed_s": round(elapsed, 2), "ts": time.time(),
        })
        print(f"  [{i}/{len(rows)}] {row['composer']} - {row['title']} "
              f"(documented: {documented_key}, fame={row['fame_tier']}): "
              f"hint={key_mentioned_hint} ({elapsed:.1f}s)")
        print(f"      raw: {response!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    print("NEXT: read every raw_response by hand before citing any number -- "
          "key_mentioned_hint is a substring heuristic, not a real scorer.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap rows, for a smoke test")
    ap.add_argument("--out", type=Path,
                    default=RESULTS_ROOT / "external_benchmarks" / "real_recordings_qwen25omni.json")
    a = ap.parse_args()
    run(a.out, a.limit)
