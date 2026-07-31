"""Track H — IN-AUDIO reference tone for tuning_judgment (novel track,
2026-07-31). Track D-zoom needed a rendered image with an explicit reference
line to fix absolute tuning; Track E's plain text value wasn't enough (no
reference to compare against). This tests whether the same "give it a
reference" ingredient works withOUT switching modality at all — mix a
reference tone into the audio itself (musicprobe/reftone.py), the way a
musician tunes against a reference note.

Same discipline as Track D/G: dropout-style training (plain / reftone mixed,
never wrong_reftone) so both eval conditions stay in-distribution, held-out
split (train_track_c._held_out_mask — tuning_judgment has no soundfont
factor, uses the base_midi top-quintile fallback), paired McNemar eval over
3 seeds, wrong_reftone mechanism control (does a WRONG reference mislead the
model, confirming real comparison rather than "two tones present" alone?).

  python gpu/train_track_h_reftone.py --seed 0 --smoke-test
  python gpu/train_track_h_reftone.py --seed 0
  python gpu/train_track_h_reftone.py --seed 0 --eval-only

Prereqs (CPU-only, run once, already done + committed 2026-07-31):
  python scripts/render_reftones.py            # 120 stimuli x 2 new WAV variants
  python -m musicprobe.reftone_jobs            # manifests/reftone_jobs.parquet

Eval writes responses__qwen25omni-reftone-s{seed}.parquet; analyze with
gpu/analyze_track_h.py.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
from train_track_d import load_qwen_omni_for_training, build_lora_config  # noqa: E402
from train_track_c import assert_lora_applied, _held_out_mask  # noqa: E402
from musicprobe.config import EXP_ROOT, RESULTS_DIR, MANIFEST_PATH, MANIFEST_DIR  # noqa: E402
from musicprobe.reftone_jobs import REFTONE_JOBS_PATH, build_reftone_jobs, _save  # noqa: E402

# training-example modality mix (per step): plain / reftone. wrong_reftone is
# NEVER trained on -- it's an eval-only mechanism control, same convention as
# wrong_image/wrong_audio elsewhere in this project.
MODE_P = {"plain": 0.5, "reftone": 0.5}


def _build_example(processor, exp_root, prompt, audio_path, answer):
    import torch
    content = [{"type": "audio", "audio": str(exp_root / audio_path)},
               {"type": "text", "text": prompt}]
    conv = [{"role": "user", "content": content}]
    pin = processor.apply_chat_template(conv, add_generation_prompt=True,
                                        tokenize=True, return_dict=True, return_tensors="pt")
    plen = pin["input_ids"].shape[1]
    ans = processor.tokenizer(answer + processor.tokenizer.eos_token,
                              return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = torch.cat([pin["input_ids"], ans], dim=1)
    labels = input_ids.clone(); labels[:, :plen] = -100
    out = dict(pin); out["input_ids"] = input_ids; out["labels"] = labels
    if "attention_mask" in out:
        out["attention_mask"] = torch.cat([out["attention_mask"], torch.ones_like(ans)], dim=1)
    return out


class DropoutDataset:
    """Per-example draw between the plain and reftone AUDIO CLIPS for the same
    underlying stimulus (both rows already exist in reftone_jobs.parquet —
    this just samples which row to use each time, no on-the-fly audio work)."""
    def __init__(self, plain_rows, reftone_rows, processor, exp_root, seed):
        assert len(plain_rows) == len(reftone_rows)
        self.plain_rows = plain_rows.reset_index(drop=True)
        self.reftone_rows = reftone_rows.reset_index(drop=True)
        self.processor, self.exp_root = processor, exp_root
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.plain_rows)

    def __getitem__(self, i):
        use_reftone = self.rng.random() < MODE_P["reftone"]
        r = self.reftone_rows.iloc[i] if use_reftone else self.plain_rows.iloc[i]
        return _build_example(self.processor, self.exp_root, r.prompt, r.audio_path, r.ground_truth)


def _assert_files_exist(jobs: pd.DataFrame, exp_root: Path):
    """reftone_jobs.parquet is committed to git, but the WAV files it points at
    are NOT (stimuli/ is gitignored). Loading an already-committed parquet
    skips the _save() rebuild below, so without this check a missing-render
    mistake fails deep inside a training or eval loop instead of immediately."""
    missing = [p for p in jobs["audio_path"].dropna().unique() if not (exp_root / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} reftone audio file(s) referenced by reftone_jobs.parquet "
            f"don't exist on this box (e.g. {missing[0]}) — run "
            "`python scripts/render_reftones.py` before training.")


def _split(exp_root):
    if not REFTONE_JOBS_PATH.exists():
        _save(build_reftone_jobs())
    jobs = pd.read_parquet(REFTONE_JOBS_PATH)
    _assert_files_exist(jobs, exp_root)
    man = pd.read_parquet(MANIFEST_PATH)[["stimulus_id", "factors"]]
    wf = jobs.merge(man, on="stimulus_id", how="left")
    ho = _held_out_mask(wf).values
    train = jobs[~ho]
    held = jobs[ho]  # all 3 conditions, eval
    train_plain = train[train.reftone_condition == "plain"].sort_values("stimulus_id")
    train_reftone = train[train.reftone_condition == "reftone"].sort_values("stimulus_id")
    return train_plain.reset_index(drop=True), train_reftone.reset_index(drop=True), \
        held.reset_index(drop=True)


def tag(seed):
    return f"qwen25omni-reftone-s{seed}"


def train(seed, smoke, exp_root):
    import torch
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments
    torch.manual_seed(seed)
    train_plain, train_reftone, held = _split(exp_root)
    print(f"[reftone] seed={seed}: {len(train_plain)} stimuli (dropout {MODE_P}) "
          f"/ {len(held)} held-out job rows")
    model, processor, lm_path = load_qwen_omni_for_training()
    model.thinker = get_peft_model(model.thinker, build_lora_config(lm_path))
    assert_lora_applied(model.thinker, f"reftone-s{seed}")
    model.thinker.train()
    if smoke:
        train_plain, train_reftone = train_plain.head(8), train_reftone.head(8)
    ds = DropoutDataset(train_plain, train_reftone, processor, exp_root, seed)
    out_dir = GPU_DIR / "track_h_reftone_ckpt" / tag(seed)
    args = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if smoke else 8,
        num_train_epochs=1 if smoke else 3, max_steps=8 if smoke else -1,
        learning_rate=2e-4, bf16=True, seed=seed,
        logging_steps=1 if smoke else 10, save_strategy="no", report_to=[],
        remove_unused_columns=False)
    Trainer(model=model.thinker, args=args, train_dataset=ds,
            data_collator=lambda b: b[0]).train()
    if smoke:
        print("[reftone] smoke done — inspect loss, then rerun without --smoke-test")
        return None, None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    model.thinker.save_pretrained(str(out_dir))
    print(f"[reftone] adapter saved to {out_dir}")
    return model, processor, held


def load_for_eval(seed):
    from peft import PeftModel
    base, processor, _ = load_qwen_omni_for_training()
    base.thinker = PeftModel.from_pretrained(
        base.thinker, str(GPU_DIR / "track_h_reftone_ckpt" / tag(seed)))
    _, _, held = _split(EXP_ROOT)
    return base, processor, held


def evaluate(seed, model, processor, held, exp_root):
    import torch
    model.eval()
    results = []
    for row in held.itertuples():
        content = [{"type": "audio", "audio": str(exp_root / row.audio_path)},
                   {"type": "text", "text": row.prompt}]
        inp = processor.apply_chat_template([{"role": "user", "content": content}],
                                            add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=64, do_sample=False, return_audio=False)
        if isinstance(out, tuple):
            out = out[0]
        out = out[:, inp["input_ids"].shape[1]:]
        raw = processor.batch_decode(out, skip_special_tokens=True)[0]
        results.append({"job_id": row.job_id, "stimulus_id": row.stimulus_id,
                        "task": row.task, "reftone_condition": row.reftone_condition,
                        "model": tag(seed), "seed": seed, "raw_response": raw,
                        "error": None, "ts": time.time()})
    out_path = RESULTS_DIR / f"responses__{tag(seed)}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"[reftone] seed={seed}: {len(results)} responses -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--exp-root", default=str(EXP_ROOT))
    a = ap.parse_args()
    exp_root = Path(a.exp_root)
    if a.eval_only:
        m, p, h = load_for_eval(a.seed); evaluate(a.seed, m, p, h, exp_root)
    else:
        m, p, h = train(a.seed, a.smoke_test, exp_root)
        if m is not None:
            evaluate(a.seed, m, p, h, exp_root)
