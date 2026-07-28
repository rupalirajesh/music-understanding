"""Track D, conclusive version: does adding a spectrogram image GENUINELY help,
free of the confounds in the first pass (gpu/train_track_d.py)?

The first pass trained on the `image` condition ONLY, so its `no_image` eval was
OUT-OF-DISTRIBUTION -> the image-vs-no_image gap conflated "image adds info" with
"model trained-with-image degrades when the image is removed." This version fixes
that and adds the statistics that were missing:

  * MIXED-CONDITION TRAINING: train ONE model on a 50/50 mix of `image` +
    `no_image` rows (both correct/non-contradictory; NEVER wrong_image, which
    would teach the model to distrust images). Both conditions are now
    in-distribution, so image-vs-no_image on the held-out set is a clean,
    WITHIN-MODEL, SAME-STIMULUS paired comparison.
  * EVAL on the held-out stimuli under all 4 conditions:
      image             correct audio + correct spectrogram
      no_image          correct audio only            (paired baseline)
      wrong_image       correct audio + wrong image   (uses image CONTENT?)
      image_wrong_audio wrong audio + correct image   (SUBSTITUTE vs COMPLEMENT)
  * MULTIPLE SEEDS (run once per --seed); analyze_track_d.py aggregates with a
    paired McNemar test + bootstrap CI across seeds.

  python gpu/train_track_d_conclusive.py --seed 0 --smoke-test
  python gpu/train_track_d_conclusive.py --seed 0
  python gpu/train_track_d_conclusive.py --seed 0 --eval-only

Same thinker-wrapping + float-dtype discipline as the fixed train_track_d.py.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR))
sys.path.insert(0, str(GPU_DIR.parent))
from train_track_d import load_qwen_omni_for_training, build_lora_config  # noqa: E402
from train_track_c import assert_lora_applied, _held_out_mask  # noqa: E402
from musicprobe.config import EXP_ROOT, RESULTS_DIR, MANIFEST_PATH  # noqa: E402
from musicprobe.image_jobs import IMAGE_JOBS_PATH, DEFAULT_TASKS, build_image_jobs  # noqa: E402
from musicprobe.scoring import parse_response, is_correct  # noqa: E402

TRAIN_CONDITIONS = ["image", "no_image"]  # non-contradictory only


def _build_example(processor, exp_root, prompt, audio_path, image_path, answer):
    """Prompt/answer-masked example; INCLUDES the image only when present
    (image_path is None for no_image rows -> audio-only example)."""
    import torch
    content = [{"type": "audio", "audio": str(exp_root / audio_path)}]
    if isinstance(image_path, str):
        content.append({"type": "image", "image": str(exp_root / image_path)})
    content.append({"type": "text", "text": prompt})
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


class MixedDataset:
    def __init__(self, rows, processor, exp_root):
        self.rows, self.processor, self.exp_root = rows, processor, exp_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        return _build_example(self.processor, self.exp_root, r.prompt,
                              r.audio_path, r.image_path, r.ground_truth)


def _split(exp_root):
    if not IMAGE_JOBS_PATH.exists():
        build_image_jobs()
    jobs = pd.read_parquet(IMAGE_JOBS_PATH)
    man = pd.read_parquet(MANIFEST_PATH)[["stimulus_id", "factors"]]
    wf = jobs.merge(man, on="stimulus_id", how="left")
    ho = _held_out_mask(wf).values
    train = jobs[(~ho) & jobs.image_condition.isin(TRAIN_CONDITIONS)]
    held = jobs[ho]  # all 4 conditions, for eval
    for t in DEFAULT_TASKS:
        assert (train.task == t).sum() > 0, f"task={t}: 0 training rows"
        assert (held.task == t).sum() > 0, f"task={t}: 0 held-out rows"
    return train.reset_index(drop=True), held.reset_index(drop=True)


def model_tag(seed):
    return f"qwen25omni-lora-mix-s{seed}"


def train(seed, smoke_test, exp_root):
    import torch
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments

    torch.manual_seed(seed)
    train_rows, held = _split(exp_root)
    train_rows = train_rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"[trackd-conc] seed={seed}: {len(train_rows)} train "
          f"(image+no_image mix) / {len(held)} held-out (4 conditions)")

    model, processor, lm_path = load_qwen_omni_for_training()
    cfg = build_lora_config(lm_path)
    model.thinker = get_peft_model(model.thinker, cfg)
    assert_lora_applied(model.thinker, f"mix-s{seed}")
    model.thinker.train()

    ds = MixedDataset(train_rows.head(8) if smoke_test else train_rows, processor, exp_root)
    out_dir = GPU_DIR / "track_d_conc_checkpoints" / model_tag(seed)
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if smoke_test else 8,
        num_train_epochs=1 if smoke_test else 3,
        max_steps=8 if smoke_test else -1,
        learning_rate=2e-4, bf16=True, seed=seed,
        logging_steps=1 if smoke_test else 10,
        save_strategy="no", report_to=[], remove_unused_columns=False,
    )
    trainer = Trainer(model=model.thinker, args=args, train_dataset=ds,
                      data_collator=lambda batch: batch[0])
    trainer.train()
    if smoke_test:
        print("[trackd-conc] smoke done — inspect loss, then rerun without --smoke-test")
        return None, None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    model.thinker.save_pretrained(str(out_dir))
    print(f"[trackd-conc] adapter saved to {out_dir}")
    return model, processor, held


def load_for_eval(seed):
    from peft import PeftModel
    base, processor, _ = load_qwen_omni_for_training()
    base.thinker = PeftModel.from_pretrained(
        base.thinker, str(GPU_DIR / "track_d_conc_checkpoints" / model_tag(seed)))
    _, held = _split(EXP_ROOT)
    return base, processor, held


def evaluate(seed, model, processor, held, exp_root):
    import torch
    model.eval()
    results = []
    for row in held.itertuples():
        content = [{"type": "audio", "audio": str(exp_root / row.audio_path)}]
        if isinstance(row.image_path, str):
            content.append({"type": "image", "image": str(exp_root / row.image_path)})
        content.append({"type": "text", "text": row.prompt})
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                 return_audio=False)
        if isinstance(out, tuple):
            out = out[0]
        out = out[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(out, skip_special_tokens=True)[0]
        results.append({"job_id": row.job_id, "stimulus_id": row.stimulus_id,
                        "task": row.task, "image_condition": row.image_condition,
                        "model": model_tag(seed), "seed": seed,
                        "raw_response": raw, "error": None, "ts": time.time()})
    out_path = RESULTS_DIR / f"responses__{model_tag(seed)}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"[trackd-conc] seed={seed}: {len(results)} responses -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--exp-root", default=str(EXP_ROOT))
    args = ap.parse_args()
    exp_root = Path(args.exp_root)
    if args.eval_only:
        model, processor, held = load_for_eval(args.seed)
        evaluate(args.seed, model, processor, held, exp_root)
    else:
        model, processor, held = train(args.seed, args.smoke_test, exp_root)
        if model is not None:
            evaluate(args.seed, model, processor, held, exp_root)
