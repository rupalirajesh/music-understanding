"""Track Z -- the auxiliary self-transcription training objective
(RESEARCH_PLAN.md Sec12.3, PROJECT_STATE.md next action 17). The one
intervention in this project where the model practices producing its OWN
intermediate representation during training, instead of reading one we
inject at test time (contrast every image/text front-end track, D-Y).

Multi-task LoRA fine-tune on Qwen2.5-Omni-7B, same audio+prompt->answer SFT
recipe as every other track here, with each training step randomly assigned
one of two objectives on the SAME audio clip:
  answer      (weight 0.6)  the normal battery question -> ground_truth
  transcribe  (weight 0.4)  a fixed transcription prompt -> the audio-derived
                             JSON event-list target (musicprobe.transcription_
                             target, format decided 2026-08-12)
At test time only the answer objective is ever used (see evaluate() below) --
the transcribe head is discarded, exactly as Sec12.3 specifies. A real
before/after read needs L2 probes (gpu/probe.py, or gpu/probe.py --mlp per
Track next action 19) on this checkpoint's own-encoder activations, compared
against the existing pre-fine-tune numbers -- not implemented in this file,
follow-up step once a checkpoint exists.

Runs across ALL THIRTEEN tasks in the frozen battery (not one cluster), per
Sec12.3's decision -- unlike every track before it, which targeted one
cluster (pitch: C-F; harmony: G/L-Q/X; rhythm: H/R-W/Y). Split logic is
therefore reused from image_track_common's 3-tier _held_out_mask (soundfont
-> base_midi -> bpm-quantile), the only one of the three held-out functions
in this codebase that's confirmed (2026-08-12, this file's CPU-side
verification) to produce non-empty train/held splits for every task in the
full jobs table, not just harmony/rhythm.

CPU-side verified 2026-08-12 (laptop, no GPU needed):
  - split() produces non-empty train/held for all 13 tasks (see PROJECT_STATE
    next action 17 for the exact counts).
  - every audio_path referenced by the split is present in
    manifests/transcription_target.json (0 missing).
GPU steps (train/evaluate below) are UNVERIFIED -- no GPU on this laptop,
same status as every other not-yet-run track's train()/evaluate() pair.

  python gpu/train_track_z_transcribe.py --seed 0 --smoke-test
  python gpu/train_track_z_transcribe.py --seed 0
  python gpu/train_track_z_transcribe.py --seed 0 --eval-only
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
from train_track_c import assert_lora_applied  # noqa: E402
from image_track_common import _held_out_mask  # noqa: E402
from musicprobe.config import EXP_ROOT, RESULTS_DIR, MANIFEST_PATH  # noqa: E402
from musicprobe.jobs import JOBS_PATH  # noqa: E402
from musicprobe.transcription_target import load_map  # noqa: E402

MODE_P = {"answer": 0.6, "transcribe": 0.4}
TRANSCRIBE_PROMPT = (
    "Transcribe what you hear. List each note you detect as a JSON object with "
    "onset (seconds), dur (seconds), and hz (fundamental frequency), in time "
    "order, as a single JSON array. If you detect no clear notes, respond []."
)
TMAP = None   # audio_path -> transcription-target JSON text


def _init_maps():
    global TMAP
    if TMAP is None:
        TMAP = load_map()


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


class MultiTaskDataset:
    def __init__(self, rows, processor, exp_root, seed):
        self.rows, self.processor, self.exp_root = rows, processor, exp_root
        self.rng = np.random.default_rng(seed)
        self.modes = list(MODE_P); self.probs = [MODE_P[m] for m in self.modes]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        mode = self.modes[int(self.rng.choice(len(self.modes), p=self.probs))]
        if mode == "transcribe":
            prompt, answer = TRANSCRIBE_PROMPT, TMAP.get(r.audio_path, "[]")
        else:
            prompt, answer = r.prompt, r.ground_truth
        return _build_example(self.processor, self.exp_root, prompt, r.audio_path, answer)


def tag(seed):
    return f"qwen25omni-transcribe-s{seed}"


def split(exp_root: Path, jobs_path=JOBS_PATH, manifest_path=MANIFEST_PATH, tasks=None):
    """Mirrors train_track_c._load_split's shape, generalized to every task
    (tasks=None) via image_track_common's 3-tier _held_out_mask -- the only
    held-out function in this codebase verified to work outside
    harmony/rhythm too (see module docstring)."""
    jobs = pd.read_parquet(jobs_path)
    man = pd.read_parquet(manifest_path)[["stimulus_id", "factors"]]
    sub = jobs[jobs["condition"] == "audio"]
    if tasks is not None:
        sub = sub[sub["task"].isin(tasks)]
    sub = sub.merge(man, on="stimulus_id", how="left")
    ho = _held_out_mask(sub)
    train, held = sub[~ho], sub[ho]
    for t in sorted(sub["task"].unique()):
        n_ho = (held["task"] == t).sum()
        assert n_ho > 0, (
            f"task={t}: held-out split produced 0 rows -- extend _held_out_mask "
            "before training on it (see image_track_common.py).")
    return train.reset_index(drop=True), held.reset_index(drop=True)


def train(seed, smoke, exp_root):
    import torch
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments
    torch.manual_seed(seed)
    _init_maps()
    train_rows, held = split(exp_root)
    print(f"[transcribe] seed={seed}: {len(train_rows)} rows (task-dropout {MODE_P}) "
          f"/ {len(held)} held-out, {train_rows.task.nunique()} tasks")
    model, processor, lm_path = load_qwen_omni_for_training()
    model.thinker = get_peft_model(model.thinker, build_lora_config(lm_path))
    assert_lora_applied(model.thinker, f"transcribe-s{seed}")
    model.thinker.train()
    ds = MultiTaskDataset(train_rows.head(8) if smoke else train_rows, processor, exp_root, seed)
    out_dir = GPU_DIR / "track_z_ckpt" / tag(seed)
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
        print("[transcribe] smoke done -- inspect loss, then rerun without --smoke-test")
        return None, None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    model.thinker.save_pretrained(str(out_dir))
    print(f"[transcribe] adapter saved to {out_dir}")
    return model, processor, held


def load_for_eval(seed):
    from peft import PeftModel
    base, processor, _ = load_qwen_omni_for_training()
    base.thinker = PeftModel.from_pretrained(base.thinker, str(GPU_DIR / "track_z_ckpt" / tag(seed)))
    _, held = split(EXP_ROOT)
    return base, processor, held


def evaluate(seed, model, processor, held, exp_root):
    """Answer-objective only -- the transcribe head is discarded at test
    time, per Sec12.3 ('re-run the existing battery' on the resulting
    encoder, not a transcription eval)."""
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
                        "task": row.task, "model": tag(seed), "seed": seed,
                        "raw_response": raw, "error": None, "ts": time.time()})
    out_path = RESULTS_DIR / f"responses__{tag(seed)}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"[transcribe] seed={seed}: {len(results)} responses -> {out_path}")
    print("[transcribe] NEXT: re-run gpu/extract_activations.py --own-encoder on this "
          "checkpoint and gpu/probe.py (or --mlp, next action 19) to compare L2 probe "
          "accuracy against the pre-fine-tune baseline -- that comparison, not this "
          "battery score alone, is Sec12.3's actual deliverable.")


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
