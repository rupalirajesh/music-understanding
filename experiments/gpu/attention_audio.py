"""Track B diagnostic (H100): how much attention does the LM actually pay to
the audio tokens — and does it decay as generation proceeds?

Tests the "the encoder hears but the LM ignores it" hypothesis directly:
linear probes (probe.py) show whether the information EXISTS in the encoder;
this shows whether the language model ROUTES any of it into its answer.

  python gpu/attention_audio.py --model Qwen/Qwen2-Audio-7B-Instruct

Per sampled job x generation step x layer, records the fraction of attention
mass on the audio token positions, alongside the uniform-attention baseline
(n_audio_tokens / context_len). Reading the output:
  attn_audio_frac >> uniform_frac  -> the LM is looking at the audio
  attn_audio_frac ~= uniform_frac  -> audio tokens are furniture
  decays across steps              -> listens at first token, then answers
                                      from its own momentum

Writes results/trackB/attention/:
  attn__<model>.parquet       long format: job_id, task, step, layer, fracs
  attn_summary__<model>.csv   per task: mean frac by layer + by step (decay curve)

Open-weights models only (APIs don't expose attention). Qwen2-Audio works with
the loader below; add a prepare() per additional model family.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musicprobe.config import TRACKB_DIR, EXP_ROOT
from musicprobe.jobs import JOBS_PATH

PER_TASK = 6          # sampled audio-condition MCQ jobs per task
MAX_NEW_TOKENS = 24   # enough steps to see the decay


def prepare_qwen2_audio(model_name: str):
    import torch
    import soundfile as sf
    import librosa
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    # eager attention: sdpa/flash don't return attention weights
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    target_sr = processor.feature_extractor.sampling_rate
    audio_token_id = getattr(model.config, "audio_token_index",
                             getattr(model.config, "audio_token_id", None))
    assert audio_token_id is not None, "can't find audio token id in config"

    def encode(prompt: str, audio_path: str):
        y, sr = sf.read(EXP_ROOT / audio_path)
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "stimulus.wav"},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(conversation,
                                             add_generation_prompt=True,
                                             tokenize=False)
        inputs = processor(text=text, audio=[y], sampling_rate=target_sr,
                           return_tensors="pt", padding=True).to("cuda")
        return inputs, audio_token_id

    return model, processor, encode


def main(model_name: str, per_task: int, seed: int = 0):
    import torch

    model, processor, encode = prepare_qwen2_audio(model_name)
    jobs = pd.read_parquet(JOBS_PATH)
    jobs = jobs[(jobs["condition"] == "audio") & (jobs["format"] == "mcq")]
    sample = (jobs.groupby("task", group_keys=False)
                  .apply(lambda g: g.sample(min(per_task, len(g)),
                                            random_state=seed)))
    print(f"[attn] {model_name}: {len(sample)} jobs "
          f"({sample['task'].nunique()} tasks x <= {per_task})")

    rows = []
    for n, job in enumerate(sample.itertuples(), 1):
        inputs, audio_token_id = encode(job.prompt, job.audio_path)
        # audio positions in the LM context = where the expanded audio
        # placeholder tokens sit in input_ids
        audio_mask = (inputs.input_ids[0] == audio_token_id)
        n_audio = int(audio_mask.sum())
        if n_audio == 0:
            continue
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, output_attentions=True,
                                 return_dict_in_generate=True)
        answer = processor.batch_decode(
            out.sequences[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True)[0]
        # out.attentions: per generated step, per layer,
        # (batch, heads, q_len, k_len). Step 0 is prefill (q = whole prompt,
        # take the last query position = the token that starts the answer).
        for step, layers in enumerate(out.attentions):
            k_len = layers[0].shape[-1]
            for li, att in enumerate(layers):
                a = att[0, :, -1, :]                        # (heads, k_len)
                frac = a[:, audio_mask[:k_len]].sum(-1).mean()
                rows.append({"model": model_name, "job_id": job.job_id,
                             "task": job.task, "answer": answer,
                             "step": step, "layer": li,
                             "attn_audio_frac": float(frac),
                             "uniform_frac": n_audio / k_len,
                             "n_audio_tokens": n_audio})
        if n % 10 == 0:
            print(f"  {n}/{len(sample)}")

    df = pd.DataFrame(rows)
    out_dir = TRACKB_DIR / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = model_name.replace("/", "_")
    df.to_parquet(out_dir / f"attn__{tag}.parquet", index=False)

    by_layer = (df.groupby(["task", "layer"])[["attn_audio_frac", "uniform_frac"]]
                  .mean().reset_index().assign(view="by_layer"))
    by_step = (df.groupby(["task", "step"])[["attn_audio_frac", "uniform_frac"]]
                 .mean().reset_index().assign(view="by_step"))
    summary = pd.concat([by_layer, by_step], ignore_index=True)
    summary.to_csv(out_dir / f"attn_summary__{tag}.csv", index=False)
    print(f"wrote {out_dir}/attn__{tag}.parquet + attn_summary__{tag}.csv")
    print("\ndecay preview (mean attn on audio by step, all tasks):")
    print(df.groupby("step")["attn_audio_frac"].mean().round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--per-task", type=int, default=PER_TASK)
    args = ap.parse_args()
    main(args.model, args.per_task)
