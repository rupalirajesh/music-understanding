"""Run eval jobs against OPEN-WEIGHTS audio LLMs on a local GPU (H100).

Ship the whole experiments/ directory to the GPU box, then:

  pip install torch transformers accelerate soundfile pandas pyarrow librosa
  python -m musicprobe.runners.run_local --model Qwen/Qwen2-Audio-7B-Instruct
  python -m musicprobe.scoring --model Qwen/Qwen2-Audio-7B-Instruct

Same jobs file, same resumable results format as run_api.py, so scoring and
review exports work identically for API and local models.

Model roster (plan §3.5) and loader status:
  Qwen/Qwen2-Audio-7B-Instruct   ✅ implemented below (the harness-validation
                                    anchor: replicate its published MMAU-music
                                    number before trusting anything else)
  Qwen2.5-Omni-7B                TODO: transformers Qwen2_5OmniForConditionalGeneration
  nvidia/audio-flamingo-3        TODO: needs NVIDIA's repo code (not plain HF)
  Music Flamingo                 TODO: same AF3 stack, music checkpoint
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


LOADERS = {
    "qwen2-audio": load_qwen2_audio,
}


def pick_loader(model_name: str):
    low = model_name.lower()
    if "qwen2-audio" in low:
        return LOADERS["qwen2-audio"]
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
            raw, err = generate(row.prompt, row.audio_path, mt), None
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
