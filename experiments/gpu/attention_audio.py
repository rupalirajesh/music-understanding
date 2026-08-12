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

Open-weights models only (APIs don't expose attention). Model roster and
loader status (mirrors musicprobe/runners/run_local.py's roster comment):
  Qwen/Qwen2-Audio-7B-Instruct       done, trusted (ran before the eager-
                                      attention check below existed, but its
                                      numbers look structurally sane: real
                                      per-layer/per-task variation, not flat)
  Qwen/Qwen2.5-Omni-7B               ran 2026-07-24, but WITHOUT the eager-
  Qwen/Qwen3-Omni-30B-A3B-Instruct   attention check below (added after the
  nvidia/audio-flamingo-3-hf         fact) -- some transformers versions
  nvidia/music-flamingo-2601-hf      silently fall back to sdpa, which would
                                      make output_attentions wrong/absent
                                      without erroring. Results for these four
                                      showed suspiciously flat, near-identical-
                                      shaped low attention across very
                                      different architectures -- re-run ALL
                                      FOUR before trusting attn_summary again;
                                      assert_eager_attention() now hard-fails
                                      immediately if this happens again.
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


def prepare_qwen_omni(model_name: str):
    """Qwen2.5-Omni / Qwen3-Omni, thinker-only (talker disabled). UNVERIFIED —
    same audio_token_id auto-detection as Qwen2-Audio; if that assert fires,
    print(model.config) and look for whichever field holds it on this
    checkpoint (naming has moved around across Qwen-Omni releases)."""
    import torch
    import soundfile as sf
    import librosa
    from transformers import AutoProcessor
    if "qwen3" in model_name.lower():
        from transformers import Qwen3OmniMoeForConditionalGeneration as Cls
    else:
        from transformers import Qwen2_5OmniForConditionalGeneration as Cls

    processor = AutoProcessor.from_pretrained(model_name)
    model = Cls.from_pretrained(model_name, torch_dtype="auto", device_map="auto",
                                attn_implementation="eager").eval()
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    target_sr = 16000
    # Qwen-Omni nests it in thinker_config, and the field name differs by version:
    # 2.5-Omni -> thinker_config.audio_token_index; 3-Omni -> thinker_config.audio_token_id
    def _atk(cfg):
        for f in ("audio_token_id", "audio_token_index"):
            v = getattr(cfg, f, None)
            if v is not None:
                return v
        return None
    tc = getattr(model.config, "thinker_config", None)
    audio_token_id = (_atk(tc) if tc is not None else None)
    if audio_token_id is None:
        audio_token_id = _atk(model.config)
    assert audio_token_id is not None, "can't find audio token id in config"

    def encode(prompt: str, audio_path: str):
        y, sr = sf.read(EXP_ROOT / audio_path)
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio": "stimulus.wav"},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(conversation,
                                             add_generation_prompt=True,
                                             tokenize=False)
        inputs = processor(text=text, audio=[y], sampling_rate=target_sr,
                           return_tensors="pt", padding=True).to(model.device)
        for k in list(inputs.keys()):
            if torch.is_floating_point(inputs[k]):
                inputs[k] = inputs[k].to(model.dtype)
        return inputs, audio_token_id

    return model, processor, encode


def prepare_audio_flamingo(model_name: str):
    """Audio Flamingo 3 / Music Flamingo, transformers-native "-hf" checkpoints.
    UNVERIFIED — audio_token_index name guessed from the Qwen2-Audio/Omni
    convention; if the assert fires, inspect model.config for the real field."""
    import torch
    from transformers import AutoProcessor
    if "music-flamingo" in model_name.lower():
        from transformers import MusicFlamingoForConditionalGeneration as Cls
    else:
        from transformers import AudioFlamingo3ForConditionalGeneration as Cls

    processor = AutoProcessor.from_pretrained(model_name)
    model = Cls.from_pretrained(model_name, torch_dtype=torch.float32,
                                device_map="cuda",
                                attn_implementation="eager").eval()
    audio_token_id = getattr(model.config, "audio_token_index",
                             getattr(model.config, "audio_token_id", None))
    assert audio_token_id is not None, "can't find audio token id in config"

    def encode(prompt: str, audio_path: str):
        content = [{"type": "audio", "path": str(EXP_ROOT / audio_path)},
                   {"type": "text", "text": prompt}]
        conversation = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        return inputs, audio_token_id

    return model, processor, encode


PREPARERS = {
    "qwen2-audio": prepare_qwen2_audio,
    "qwen-omni": prepare_qwen_omni,
    "flamingo": prepare_audio_flamingo,
}


def _resolved_attn_impls(model) -> dict[str, str | None]:
    """Every place a resolved attn_implementation could live: top-level config,
    and any nested sub-configs (Qwen-Omni nests it under thinker_config /
    audio_config / text_config depending on version). HF sets this on
    `_attn_implementation` at each level that actually has its own attention
    modules — if any of them didn't land on 'eager', output_attentions for
    those layers is silently wrong or absent, which is exactly the failure
    mode this whole diagnostic is vulnerable to (see module docstring)."""
    found = {}
    cfg = model.config
    found["top"] = getattr(cfg, "_attn_implementation", None)
    for sub in ("thinker_config", "text_config", "audio_config",
                "language_model_config", "talker_config"):
        sub_cfg = getattr(cfg, sub, None)
        if sub_cfg is not None:
            found[sub] = getattr(sub_cfg, "_attn_implementation", None)
    return found


def assert_eager_attention(model, model_name: str) -> None:
    """Hard-fail if eager attention didn't actually take effect anywhere in
    the model. Silently proceeding here is how the previous run produced
    attn_summary numbers nobody could actually trust (see 2026-07-24 report
    correction) -- do not soften this back to a warning."""
    impls = _resolved_attn_impls(model)
    bad = {k: v for k, v in impls.items() if v is not None and v != "eager"}
    print(f"[attn] {model_name}: resolved attn_implementation = {impls}")
    assert not bad, (
        f"{model_name}: attn_implementation did NOT resolve to 'eager' for "
        f"{bad} -- output_attentions will be missing/wrong for those modules, "
        "and any attn_summary produced from this run is not trustworthy. "
        "Check the installed transformers version and this model's modeling "
        "code for eager-attention support before rerunning."
    )


def pick_preparer(model_name: str):
    low = model_name.lower()
    if "qwen2-audio" in low:
        return PREPARERS["qwen2-audio"]
    if "omni" in low:
        return PREPARERS["qwen-omni"]
    if "flamingo" in low:
        return PREPARERS["flamingo"]
    raise ValueError(f"no attention-diagnostic loader for {model_name} — add one above")


def main(model_name: str, per_task: int, seed: int = 0, lora_checkpoint: str = None,
        tag_override: str = None):
    """lora_checkpoint (PROJECT_STATE.md next action 22, added 2026-08-12,
    UNVERIFIED -- no GPU on the laptop to test against a real checkpoint):
    point the diagnostic at a Track L-Y-style fine-tuned adapter instead of a
    base model, to check whether images that HURT accuracy (key_id/tempo_bpm
    in Tracks L-Q/R-W) do so because attention is drawn toward the image
    tokens at audio's expense, or whether audio attention stays flat like the
    existing wrong_image~=no_image mechanism control already suggests. Only
    wired for the qwen-omni preparer (the only architecture any LoRA track in
    this project fine-tunes) -- wraps .thinker with the saved adapter the
    same way gpu/image_track_common.load_for_eval does, since that's the
    submodule Track C/D/../Z's build_lora_config actually targets."""
    import torch

    model, processor, encode = pick_preparer(model_name)(model_name)
    if lora_checkpoint is not None:
        if "omni" not in model_name.lower():
            raise ValueError("--lora-checkpoint is only wired for qwen-omni preparers "
                             "(the only architecture any LoRA track fine-tunes) -- "
                             f"got --model {model_name}")
        from peft import PeftModel
        model.thinker = PeftModel.from_pretrained(model.thinker, lora_checkpoint)
        print(f"[attn] wrapped .thinker with adapter from {lora_checkpoint}")
    assert_eager_attention(model, model_name)
    jobs = pd.read_parquet(JOBS_PATH)
    jobs = jobs[(jobs["condition"] == "audio") & (jobs["format"] == "mcq")]
    # pandas 3.0's groupby.apply drops the grouping column; iterate groups instead
    sample = pd.concat(
        [g.sample(min(per_task, len(g)), random_state=seed)
         for _, g in jobs.groupby("task")]).reset_index(drop=True)
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
        gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                          output_attentions=True, return_dict_in_generate=True)
        if "omni" in model_name.lower():   # Qwen-Omni: never synthesise speech
            gen_kwargs["return_audio"] = False
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        answer = processor.batch_decode(
            out.sequences[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True)[0]
        # out.attentions: per generated step, per layer,
        # (batch, heads, q_len, k_len). Step 0 is prefill (q = whole prompt,
        # take the last query position = the token that starts the answer).
        for step, layers in enumerate(out.attentions):
            k_len = layers[0].shape[-1]
            # align the audio mask to this step's key length: pad with False for
            # generated (non-audio) key positions, or clip to k_len
            if k_len >= audio_mask.shape[0]:
                m = torch.cat([audio_mask, torch.zeros(
                    k_len - audio_mask.shape[0], dtype=torch.bool,
                    device=audio_mask.device)])
            else:
                m = audio_mask[:k_len]
            for li, att in enumerate(layers):
                a = att[0, :, -1, :]                        # (heads, k_len)
                frac = a[:, m].sum(-1).mean()
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
    tag = tag_override or model_name.replace("/", "_")
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
    ap.add_argument("--lora-checkpoint", default=None,
                    help="path to a Track L-Y/Z-style saved adapter dir (next action 22); "
                         "only valid with --model pointing at a qwen-omni base checkpoint")
    ap.add_argument("--tag", default=None,
                    help="output filename tag override, e.g. 'track-l-chroma-picked-s0' "
                         "-- defaults to --model with / replaced by _")
    args = ap.parse_args()
    main(args.model, args.per_task, lora_checkpoint=args.lora_checkpoint, tag_override=args.tag)
