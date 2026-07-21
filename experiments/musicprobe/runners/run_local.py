"""Run eval jobs against OPEN-WEIGHTS audio LLMs on a local GPU (H100).

Ship the whole experiments/ directory to the GPU box, then:

  pip install torch transformers accelerate soundfile pandas pyarrow librosa
  python -m musicprobe.runners.run_local --model Qwen/Qwen2-Audio-7B-Instruct
  python -m musicprobe.scoring --model Qwen/Qwen2-Audio-7B-Instruct

Same jobs file, same resumable results format as run_api.py, so scoring and
review exports work identically for API and local models.

Model roster (plan §3.5) and loader status — all plain transformers (>=5.14),
no vendor forks. Checkpoint ids verified against the HF hub 2026-07-20 (all
ungated, all natively supported in transformers main):
  Qwen/Qwen2-Audio-7B-Instruct       ✅ implemented + validated (full run done)
  Qwen/Qwen2.5-Omni-7B               ✅ loader below — UNVERIFIED on hardware:
                                        smoke-test with --limit 5 first
  Qwen/Qwen3-Omni-30B-A3B-Instruct   ✅ same loader (~60GB bf16, fits one
                                        H100-80GB) — UNVERIFIED, smoke-test
  nvidia/audio-flamingo-3-hf         ✅ loader below — UNVERIFIED, smoke-test
  nvidia/music-flamingo-2601-hf      ✅ same loader (AF3 family) — UNVERIFIED
Add a loader function per model family; the job loop never changes.
"""
import argparse
import time

import pandas as pd

from ..config import RESULTS_DIR, EXP_ROOT
from ..jobs import JOBS_PATH


def load_qwen2_audio(model_name: str):
    import torch
    import soundfile as sf
    import librosa
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    target_sr = processor.feature_extractor.sampling_rate  # 16 kHz — the plan's
    # documented bottleneck: resampling happens HERE, log it, don't hide it.

    def generate(prompt: str, audio_path: str | None, max_new_tokens: int) -> str:
        content = []
        audios = None
        if isinstance(audio_path, str) and audio_path:
            y, sr = sf.read(EXP_ROOT / audio_path)
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            audios = [y]
            content.append({"type": "audio", "audio_url": "stimulus.wav"})
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True,
                                             tokenize=False)
        inputs = processor(text=text, audio=audios, sampling_rate=target_sr,
                           return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)  # greedy == temperature 0
        out = out[:, inputs.input_ids.shape[1]:]
        return processor.batch_decode(out, skip_special_tokens=True)[0]

    return generate


def load_qwen_omni(model_name: str):
    """Qwen2.5-Omni / Qwen3-Omni, thinker-only (talker disabled: text out only).
    UNVERIFIED on hardware — run with --limit 5 and eyeball responses before
    burning the full battery."""
    import torch
    import soundfile as sf
    import librosa
    from transformers import AutoProcessor
    if "qwen3" in model_name.lower():
        from transformers import Qwen3OmniMoeForConditionalGeneration as Cls
    else:
        from transformers import Qwen2_5OmniForConditionalGeneration as Cls

    processor = AutoProcessor.from_pretrained(model_name)
    model = Cls.from_pretrained(model_name, torch_dtype="auto",
                                device_map="auto").eval()
    if hasattr(model, "disable_talker"):
        model.disable_talker()  # we never want speech output; saves ~10 GB
    target_sr = 16000

    def generate(prompt: str, audio_path: str | None, max_new_tokens: int) -> str:
        content = []
        audios = None
        if isinstance(audio_path, str) and audio_path:
            y, sr = sf.read(EXP_ROOT / audio_path)
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            audios = [y]
            content.append({"type": "audio", "audio": "stimulus.wav"})
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(conversation,
                                             add_generation_prompt=True,
                                             tokenize=False)
        inputs = processor(text=text, audio=audios, sampling_rate=target_sr,
                           return_tensors="pt", padding=True).to(model.device)
        # bf16 models (Qwen3-Omni-30B) emit float32 audio features -> cast the
        # floating tensors to the model dtype IN PLACE (keep the BatchFeature so
        # inputs.input_ids still works; no-op when model is fp32/auto)
        for k in list(inputs.keys()):
            if torch.is_floating_point(inputs[k]):
                inputs[k] = inputs[k].to(model.dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, return_audio=False)
        if isinstance(out, tuple):  # some versions return (text_ids, audio)
            out = out[0]
        out = out[:, inputs.input_ids.shape[1]:]
        return processor.batch_decode(out, skip_special_tokens=True)[0]

    return generate


def load_audio_flamingo(model_name: str):
    """Audio Flamingo 3 / Music Flamingo via the transformers-NATIVE "-hf"
    checkpoints (nvidia/audio-flamingo-3-hf, nvidia/music-flamingo-2601-hf) —
    no NVIDIA vendor fork needed. The processor's chat template loads and
    resamples the audio itself from a path.
    UNVERIFIED on hardware — smoke-test with --limit 5 and eyeball responses."""
    import torch
    from transformers import AutoProcessor
    if "music-flamingo" in model_name.lower():
        from transformers import MusicFlamingoForConditionalGeneration as Cls
    else:
        from transformers import AudioFlamingo3ForConditionalGeneration as Cls

    processor = AutoProcessor.from_pretrained(model_name)
    # AF3 family has mixed-precision internals (audio tower wants float32); load
    # the whole model in float32 so every path matches (fits on an 80GB+ GPU).
    model = Cls.from_pretrained(model_name, torch_dtype=torch.float32,
                                device_map="cuda").eval()

    def generate(prompt: str, audio_path: str | None, max_new_tokens: int) -> str:
        content = []
        if isinstance(audio_path, str) and audio_path:
            content.append({"type": "audio", "path": str(EXP_ROOT / audio_path)})
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(out, skip_special_tokens=True)[0]

    return generate


LOADERS = {
    "qwen2-audio": load_qwen2_audio,
    "qwen-omni": load_qwen_omni,
    "flamingo": load_audio_flamingo,
}


def pick_loader(model_name: str):
    low = model_name.lower()
    if "qwen2-audio" in low:
        return LOADERS["qwen2-audio"]
    if "omni" in low:
        return LOADERS["qwen-omni"]
    if "flamingo" in low:
        return LOADERS["flamingo"]
    raise ValueError(f"no loader for {model_name} — add one to LOADERS")


def run(model: str, limit: int | None = None, tasks: list[str] | None = None):
    jobs = pd.read_parquet(JOBS_PATH)
    if tasks:
        jobs = jobs[jobs["task"].isin(tasks)]
    out_path = RESULTS_DIR / f"responses__{model.replace('/', '_')}.parquet"
    done, results = set(), []
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        prev = prev[prev["error"].isna()]
        done = set(prev["job_id"])
        results = prev.to_dict("records")
    todo = jobs[~jobs["job_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"[run_local] {model}: {len(todo)} jobs ({len(done)} done)")

    generate = pick_loader(model)(model)  # build the loader closure (loads model once)
    for n, row in enumerate(todo.itertuples(), 1):
        try:
            mt = 512 if row.format == "explain" else 64
            # no-audio jobs store no path; parquet round-trips None as NaN
            path = row.audio_path if isinstance(row.audio_path, str) else None
            raw, err = generate(row.prompt, path, mt), None
        except Exception as e:
            raw, err = None, f"{type(e).__name__}: {e}"
        results.append({"job_id": row.job_id, "model": model,
                        "raw_response": raw, "error": err, "ts": time.time()})
        if n % 50 == 0 or n == len(todo):
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_parquet(out_path, index=False)
            print(f"  {n}/{len(todo)}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()
    run(args.model, args.limit, args.tasks)
