"""Shared LoRA train+eval harness for the Tracks L-W image/front-end
representation ladder (harmony L/M/N/O/P/Q, rhythm R/S/T/U/V/W).

This is a DRY extraction of train_track_g_chroma.py's pattern -- modality-
dropout training (audio+image / image-only / audio-only), held-out-soundfont
split reused from Track C, same LoRA config as every Track C-H script,
paired 3-seed eval writing responses__<tag>.parquet for gpu/analyze_track_g.py-
style analysis. train_track_g_chroma.py itself is left completely untouched
(Track G already ran and its results are reported in PAPER.md -- don't risk
changing its behavior). New tracks call this module with their own
(path_fn, tasks, jobs_path, tag) instead of duplicating ~150 lines each --
one bug fix here fixes it for all twelve, instead of needing to be
independently ported into twelve near-identical files.

  from gpu.image_track_common import run_track
  run_track(seed=0, smoke=False, exp_root=EXP_ROOT, tag="qwen25omni-chroma-picked",
            tasks=("key_id","mode_id","chord_quality","interval_id"),
            image_path_fn=chroma_picked_path, jobs_path=CHROMA_PICKED_JOBS_PATH,
            ckpt_subdir="track_l_chroma_picked_ckpt")
"""
import sys
import time
from pathlib import Path

import pandas as pd

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
from train_track_d import load_qwen_omni_for_training, build_lora_config  # noqa: E402
from train_track_c import assert_lora_applied, _held_out_mask as _base_held_out_mask  # noqa: E402
from musicprobe.config import EXP_ROOT, MANIFEST_PATH, RESULTS_DIR  # noqa: E402
from musicprobe.image_jobs import build_image_jobs  # noqa: E402

MODE_P = {"both": 0.5, "image_only": 0.25, "audio_only": 0.25}


def _build_example(processor, exp_root, prompt, audio_path, image_path, answer,
                   use_audio=True, use_image=True):
    import torch
    content = []
    if use_audio and isinstance(audio_path, str):
        content.append({"type": "audio", "audio": str(exp_root / audio_path)})
    if use_image and isinstance(image_path, str):
        content.append({"type": "image", "image": str(exp_root / image_path)})
    content.append({"type": "text", "text": prompt})
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
    def __init__(self, rows, processor, exp_root, seed):
        import numpy as np
        self.rows, self.processor, self.exp_root = rows, processor, exp_root
        self.rng = np.random.default_rng(seed)
        self.modes = list(MODE_P); self.probs = [MODE_P[m] for m in self.modes]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        mode = self.modes[int(self.rng.choice(len(self.modes), p=self.probs))]
        use_audio = mode in ("both", "audio_only")
        use_image = mode in ("both", "image_only")
        return _build_example(self.processor, self.exp_root, r.prompt, r.audio_path,
                              r.image_path, r.ground_truth,
                              use_audio=use_audio, use_image=use_image)


def _assert_files_exist(jobs: pd.DataFrame, exp_root: Path, jobs_path: Path):
    missing = []
    for col in ("audio_path", "image_path"):
        paths = jobs[col].dropna().unique()
        missing += [p for p in paths if not (exp_root / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} file(s) referenced by {jobs_path.name} don't exist on this "
            f"box (e.g. {missing[0]}) -- render the representation's images before training.")


def _held_out_mask(sub: pd.DataFrame) -> pd.Series:
    """Extends train_track_c._held_out_mask's soundfont/base_midi fallback
    chain with a third tier, needed for tasks with NEITHER factor.
    tempo_bpm/beats_per_bar's factors are {bpm, meter, n_bars[, beats]} --
    no soundfont, no base_midi. Confirmed by testing (both an independent
    review pass and a direct run of split() on the real rhythm-task jobs)
    that calling train_track_c._held_out_mask on these tasks UNCHANGED
    silently returns an all-False mask: its no-soundfont branch computes
    `base_midi >= quantile(0.8)`, and base_midi is NaN for every row, so
    the comparison is False everywhere -- 0 rows held out, Tracks R-W would
    train on 100% of the data and eval on an empty set (evaluate() writes a
    0-row parquet; analyze_track_repr.py then crashes reading it back --
    caught here, before ever reaching the GPU box, not after).

    Left train_track_c._held_out_mask itself untouched (Track C/D/G/H
    already ran against it; changing its behavior risks changing already-
    published results if any of those are rerun) -- this wraps it instead,
    only overriding rows neither existing tier resolves. Fallback: held-out
    top-quintile BPM, the same "hold out the tail of a continuous factor"
    discipline as the base_midi tier, just keyed on the factor that
    actually exists here -- guards against tempo-memorization instead of
    pitch-memorization."""
    import json
    mask = _base_held_out_mask(sub)

    def parse(f):
        return json.loads(f) if isinstance(f, str) else {}
    factors = sub["factors"].apply(parse)
    has_soundfont = factors.apply(lambda d: "soundfont" in d)
    has_base_midi = factors.apply(lambda d: "base_midi" in d)
    unresolved = ~(has_soundfont | has_base_midi)
    if unresolved.any():
        bpm = pd.to_numeric(factors[unresolved].apply(lambda d: d.get("bpm")), errors="coerce")
        threshold = bpm.quantile(0.8)
        mask[unresolved] = bpm >= threshold
    return mask


def split(exp_root: Path, tasks: tuple[str, ...], image_path_fn, jobs_path: Path):
    if not jobs_path.exists():
        from musicprobe.image_jobs import _save
        _save(build_image_jobs(tasks=tasks, image_path_fn=image_path_fn), out_path=jobs_path)
    jobs = pd.read_parquet(jobs_path)
    _assert_files_exist(jobs, exp_root, jobs_path)
    man = pd.read_parquet(MANIFEST_PATH)[["stimulus_id", "factors"]]
    wf = jobs.merge(man, on="stimulus_id", how="left")
    ho = _held_out_mask(wf).values
    train = jobs[(~ho) & (jobs.image_condition == "image")]
    held = jobs[ho]
    return train.reset_index(drop=True), held.reset_index(drop=True)


def train(seed: int, smoke: bool, exp_root: Path, tasks, image_path_fn, jobs_path: Path,
         tag: str, ckpt_subdir: str):
    import torch
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments
    torch.manual_seed(seed)
    train_rows, held = split(exp_root, tasks, image_path_fn, jobs_path)
    print(f"[{tag}] seed={seed}: {len(train_rows)} image-rows (modality-dropout "
          f"{MODE_P}) / {len(held)} held-out, tasks={tasks}")
    model, processor, lm_path = load_qwen_omni_for_training()
    model.thinker = get_peft_model(model.thinker, build_lora_config(lm_path))
    assert_lora_applied(model.thinker, f"{tag}-s{seed}")
    model.thinker.train()
    ds = DropoutDataset(train_rows.head(8) if smoke else train_rows, processor, exp_root, seed)
    out_dir = GPU_DIR / ckpt_subdir / f"{tag}-s{seed}"
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
        print(f"[{tag}] smoke done -- inspect loss, then rerun without --smoke-test")
        return None, None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    model.thinker.save_pretrained(str(out_dir))
    print(f"[{tag}] adapter saved to {out_dir}")
    return model, processor, held


def load_for_eval(seed: int, tasks, image_path_fn, jobs_path: Path, tag: str, ckpt_subdir: str):
    from peft import PeftModel
    base, processor, _ = load_qwen_omni_for_training()
    base.thinker = PeftModel.from_pretrained(
        base.thinker, str(GPU_DIR / ckpt_subdir / f"{tag}-s{seed}"))
    _, held = split(EXP_ROOT, tasks, image_path_fn, jobs_path)
    return base, processor, held


def evaluate(seed: int, model, processor, held, exp_root: Path, tag: str):
    import torch
    model.eval()
    results = []
    for row in held.itertuples():
        content = []
        if isinstance(row.audio_path, str):
            content.append({"type": "audio", "audio": str(exp_root / row.audio_path)})
        if isinstance(row.image_path, str):
            content.append({"type": "image", "image": str(exp_root / row.image_path)})
        content.append({"type": "text", "text": row.prompt})
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
                        "task": row.task, "image_condition": row.image_condition,
                        "model": f"{tag}-s{seed}", "seed": seed, "raw_response": raw,
                        "error": None, "ts": time.time()})
    out_path = RESULTS_DIR / f"responses__{tag}-s{seed}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"[{tag}] seed={seed}: {len(results)} responses -> {out_path}")
    return out_path


def run_track(seed: int, smoke: bool, exp_root: Path, tasks, image_path_fn, jobs_path: Path,
              tag: str, ckpt_subdir: str, eval_only: bool = False):
    if eval_only:
        m, p, h = load_for_eval(seed, tasks, image_path_fn, jobs_path, tag, ckpt_subdir)
        evaluate(seed, m, p, h, exp_root, tag)
    else:
        m, p, h = train(seed, smoke, exp_root, tasks, image_path_fn, jobs_path, tag, ckpt_subdir)
        if m is not None:
            evaluate(seed, m, p, h, exp_root, tag)
