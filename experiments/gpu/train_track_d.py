"""Track D, Phase 1 (H100): does a spectrogram image, added alongside audio,
change accuracy on the alignment-fixable shortlist -- and is the model
actually using it, or just learning to read the image and ignore audio?
RESEARCH_PLAN.md §12.2.

  python gpu/train_track_d.py --smoke-test
  python gpu/train_track_d.py
  python gpu/train_track_d.py --eval-only     # rerun eval only, skip training

Single LoRA arm (unlike Track C's three) -- Phase 1 isn't isolating which
component needs adjusting, it's asking whether the extra modality helps at
all, and whether the wrong-image control catches a shortcut if it does.
LoRA targets the Thinker's language-model decoder only, audio/vision towers
frozen -- same reasoning as Track C's llm_only arm: cheapest test first.

Model: Qwen/Qwen2.5-Omni-7B -- the ONLY open model in this study that
accepts audio + image in the same turn (RESEARCH_PLAN.md §12.1: AF3 and
Music-Flamingo don't take image input at all; Qwen3-Omni-30B-A3B is the
same story as Track A/B -- correct choice but a heavier stretch goal, not
this first pass).

Training data: musicprobe.image_jobs' `image` condition only (correct audio
+ correct spectrogram) -- `no_image`/`wrong_image` rows are EVAL-ONLY
controls, never trained on. Held-out split reuses train_track_c.py's
_held_out_mask (same soundfont/pitch-register criterion, same reasoning).

Eval output: writes results/trackA/responses__qwen25omni-lora-image.parquet
in run_local.py's exact schema, PLUS a dedicated
results/trackA/trackd_image_summary.csv (accuracy by image_condition) since
image_condition isn't a column musicprobe.scoring's existing condition
grouping knows about -- see score_image_jobs() below, which reuses
musicprobe.scoring.parse_response/is_correct directly rather than
reimplementing answer parsing.

UNVERIFIED on hardware, same honesty as every other gpu/ script:
- LM_PATH_CANDIDATES's primary guess is "thinker.model" -- extract_activations
  .py already confirmed "thinker.audio_tower" is a direct child of thinker
  (not nested under thinker.model), which is why LoRA anchored at
  "thinker.model" should exclude the audio tower's identically-named
  q_proj/v_proj -- but the language-decoder path itself has NOT been
  confirmed against the real checkpoint. assert_lora_applied() hard-fails
  if the regex matches nothing, same discipline as train_track_c.py.
- Qwen2.5-Omni's processor image-content key is assumed to be
  {"type": "image", "image": <path>} (mirrors the already-verified audio
  key {"type": "audio", "audio": <path>} from run_local.py/
  attention_audio.py's Qwen-Omni loaders) -- not confirmed on hardware.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_track_c import _find_submodule, assert_lora_applied, _held_out_mask  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from musicprobe.scoring import parse_response, is_correct  # noqa: E402
from musicprobe.image_jobs import IMAGE_JOBS_PATH, DEFAULT_TASKS, build_image_jobs  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
LM_PATH_CANDIDATES = ["thinker.model", "thinker.language_model"]
LORA_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")


def load_qwen_omni_for_training():
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="cuda")
    if hasattr(model, "disable_talker"):
        model.disable_talker()  # text out only, saves ~10GB — same as run_local.py
    lm_path, _ = _find_submodule(model, LM_PATH_CANDIDATES,
                                 ("Qwen2Model", "Qwen2_5OmniThinkerTextModel", "CausalLM"))
    print(f"[train_track_d] language model at '{lm_path}'")
    return model, processor, lm_path


def build_lora_config(lm_path: str):
    from peft import LoraConfig

    suffix_group = "|".join(LORA_TARGET_SUFFIXES)
    pattern = rf"^{lm_path}\..*\.({suffix_group})$"
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                       task_type="CAUSAL_LM", target_modules=pattern)


# ---------------------------------------------------------------- data -----
def _load_train_and_heldout(exp_root: Path):
    if not IMAGE_JOBS_PATH.exists():
        build_image_jobs()  # image_jobs.py already asserts renders exist first
    jobs = pd.read_parquet(IMAGE_JOBS_PATH)
    man = pd.read_parquet("manifests/stimuli.parquet")[["stimulus_id", "factors"]]
    with_factors = jobs.merge(man, on="stimulus_id", how="left")
    held_out_mask = _held_out_mask(with_factors)

    train = jobs[(jobs.image_condition == "image") & ~held_out_mask.values]
    held_out = jobs[held_out_mask.values]  # all 3 image_conditions, for eval
    for t in DEFAULT_TASKS:
        assert (train.task == t).sum() > 0, f"task={t}: 0 training rows"
        assert (held_out.task == t).sum() > 0, f"task={t}: 0 held-out rows"
    return train.reset_index(drop=True), held_out.reset_index(drop=True)


def _build_example(processor, exp_root: Path, prompt: str, audio_path: str,
                   image_path: str, answer: str):
    """Same prompt/answer masking pattern as train_track_c.py's
    _build_example, extended with an image content part."""
    import torch

    content = [{"type": "audio", "audio": str(exp_root / audio_path)},
               {"type": "image", "image": str(exp_root / image_path)},
               {"type": "text", "text": prompt}]
    conversation = [{"role": "user", "content": content}]
    prompt_inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]

    answer_ids = processor.tokenizer(answer + processor.tokenizer.eos_token,
                                      return_tensors="pt",
                                      add_special_tokens=False)["input_ids"]
    input_ids = torch.cat([prompt_inputs["input_ids"], answer_ids], dim=1)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    out = dict(prompt_inputs)
    out["input_ids"] = input_ids
    out["labels"] = labels
    if "attention_mask" in out:
        out["attention_mask"] = torch.cat(
            [out["attention_mask"], torch.ones_like(answer_ids)], dim=1)
    return out


class TrackDDataset:
    def __init__(self, rows, processor, exp_root):
        self.rows, self.processor, self.exp_root = rows, processor, exp_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        return _build_example(self.processor, self.exp_root, r.prompt,
                              r.audio_path, r.image_path, r.ground_truth)


# ------------------------------------------------------------- training ----
def train(smoke_test: bool, exp_root: Path):
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments

    train_rows, held_out_rows = _load_train_and_heldout(exp_root)
    print(f"[train_track_d] {len(train_rows)} train (image-condition only) / "
          f"{len(held_out_rows)} held-out (all 3 image_conditions)")

    model, processor, lm_path = load_qwen_omni_for_training()
    cfg = build_lora_config(lm_path)
    model = get_peft_model(model, cfg)
    assert_lora_applied(model, "image")
    model.train()

    ds = TrackDDataset(train_rows.head(8) if smoke_test else train_rows, processor, exp_root)
    out_dir = Path("gpu/track_d_checkpoints/image")
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if smoke_test else 8,
        num_train_epochs=1 if smoke_test else 3,
        max_steps=8 if smoke_test else -1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1 if smoke_test else 10,
        save_strategy="no" if smoke_test else "epoch",
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=lambda batch: batch[0])
    trainer.train()

    if smoke_test:
        print("[train_track_d] smoke test done — inspect the loss above, then "
              "rerun WITHOUT --smoke-test for the full run.")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    print(f"[train_track_d] adapter saved to {out_dir}")
    return model, processor, held_out_rows


# ------------------------------------------------------------------ eval ---
def evaluate(model, processor, held_out_rows: pd.DataFrame, exp_root: Path):
    import torch

    model.eval()
    results = []
    for row in held_out_rows.itertuples():
        content = [{"type": "audio", "audio": str(exp_root / row.audio_path)}]
        if isinstance(row.image_path, str):  # None for no_image condition
            content.append({"type": "image", "image": str(exp_root / row.image_path)})
        content.append({"type": "text", "text": row.prompt})
        conversation = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                 return_audio=False)
        if isinstance(out, tuple):
            out = out[0]
        out = out[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(out, skip_special_tokens=True)[0]
        results.append({"job_id": row.job_id, "model": "qwen25omni-lora-image",
                        "raw_response": raw, "error": None, "ts": time.time()})

    out_path = Path("results/trackA/responses__qwen25omni-lora-image.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"[train_track_d] {len(results)} held-out responses -> {out_path}")
    score_image_jobs(out_path, held_out_rows)


def score_image_jobs(responses_path: Path, held_out_rows: pd.DataFrame) -> pd.DataFrame:
    """Reuses musicprobe.scoring's answer parsing (parse_response/is_correct)
    directly instead of reimplementing it -- image_condition isn't part of
    the existing scoring pipeline's condition grouping, so this writes its
    own small summary rather than forcing image_jobs through
    musicprobe.scoring.score_model(), which is hardcoded to jobs.parquet."""
    resp = pd.read_parquet(responses_path)
    df = held_out_rows.merge(resp[["job_id", "raw_response"]], on="job_id")
    df["parsed"] = [parse_response(r) for r in df.itertuples()]
    df["correct"] = [is_correct(r.task, r.parsed, r.ground_truth) for r in df.itertuples()]

    # unparseable/refused rows must score as incorrect, not get dropped from
    # the denominator (musicprobe.scoring.summary_table's own documented
    # convention, scoring.py:107-108 -- .eq(True), not .mean(skipna=True))
    summary = (df.groupby(["task", "image_condition"])["correct"]
                 .apply(lambda s: s.eq(True).mean()).reset_index())
    out_path = Path("results/trackA/trackd_image_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"[train_track_d] accuracy by task x image_condition -> {out_path}")
    print(summary.pivot(index="task", columns="image_condition", values="correct").round(3))
    return summary


def load_adapter_for_eval():
    from peft import PeftModel

    base, processor, _ = load_qwen_omni_for_training()
    model = PeftModel.from_pretrained(base, "gpu/track_d_checkpoints/image")
    return model, processor


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, load a previously-saved adapter and re-eval")
    ap.add_argument("--exp-root", default=".")
    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    if args.eval_only:
        _, held_out = _load_train_and_heldout(exp_root)
        model, processor = load_adapter_for_eval()
        evaluate(model, processor, held_out, exp_root)
    else:
        result = train(args.smoke_test, exp_root)
        if result is not None:
            model, processor, held_out_rows = result
            evaluate(model, processor, held_out_rows, exp_root)
