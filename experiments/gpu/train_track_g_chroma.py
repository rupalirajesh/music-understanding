"""Track G — chromagram front-end for the HARMONIC task cluster (novel track,
2026-07-31). Tracks C-F all targeted the pitch-precision shortlist (octave_id,
note_count, tuning_judgment, cents_discrimination). This is the first causal
test on the OTHER shortlist Track B flagged and never re-tested: key_id,
mode_id, chord_quality, interval_id (L3 > generic-encoder L2 — own-encoder
re-probe, 2026-07-24, found no clean signal there either; behavioral success
reads as priors, not perception). A chromagram (12 pitch-class rows x time,
musicprobe/chromagram.py) is the harmonic analogue of Track D's F0-contour:
same abstraction level as the audio, no answer baked in.

Same discipline as train_track_d_force.py from the start (skip Track D's
first-pass OOD mistake): MODALITY-DROPOUT training (audio+image / image-only
/ audio-only) so both eval conditions are in-distribution, wrong_image control,
held-out-soundfont split (train_track_c._held_out_mask — these 4 tasks all
have a soundfont factor, no base_midi fallback needed), paired McNemar eval
over 3 seeds (gpu/analyze_track_g.py).

  python gpu/train_track_g_chroma.py --seed 0 --smoke-test
  python gpu/train_track_g_chroma.py --seed 0
  python gpu/train_track_g_chroma.py --seed 0 --eval-only

Prereqs (CPU-only, run once, already done + committed 2026-07-31):
  python scripts/render_chromagrams.py --tasks            # whole battery (wrong-image pool)
  python -m musicprobe.image_jobs --kind chromagram        # manifests/chroma_jobs.parquet

Eval writes responses__qwen25omni-chroma-s{seed}.parquet; analyze with
gpu/analyze_track_g.py.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
from train_track_d import load_qwen_omni_for_training, build_lora_config  # noqa: E402
from train_track_c import assert_lora_applied, _held_out_mask  # noqa: E402
from musicprobe.config import EXP_ROOT, RESULTS_DIR, MANIFEST_PATH, MANIFEST_DIR  # noqa: E402
from musicprobe.image_jobs import build_image_jobs  # noqa: E402
from musicprobe.chromagram import chromagram_path  # noqa: E402

CHROMA_JOBS_PATH = MANIFEST_DIR / "chroma_jobs.parquet"
CHROMA_TASKS = ("key_id", "mode_id", "chord_quality", "interval_id")

# training-example modality mix (per step): both / image-only / audio-only
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
    """Modality-dropout over the `image` rows: audio+chromagram / image-only /
    audio-only, sampled per example."""
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


def _assert_files_exist(jobs: pd.DataFrame, exp_root: Path):
    """The jobs parquet is committed to git, but the WAV/PNG files it points at
    are NOT (stimuli/ is gitignored, deliberately regenerable). Loading an
    already-committed chroma_jobs.parquet skips the _save() rebuild below, so
    without this check a missing-render mistake fails deep inside a training
    or eval loop instead of immediately and clearly."""
    missing = []
    for col in ("audio_path", "image_path"):
        paths = jobs[col].dropna().unique()
        missing += [p for p in paths if not (exp_root / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} chromagram file(s) referenced by chroma_jobs.parquet "
            f"don't exist on this box (e.g. {missing[0]}) — run "
            "`python scripts/render_chromagrams.py --tasks` before training.")


def _split(exp_root):
    if not CHROMA_JOBS_PATH.exists():
        from musicprobe.image_jobs import _save
        _save(build_image_jobs(tasks=CHROMA_TASKS, image_path_fn=chromagram_path),
              out_path=CHROMA_JOBS_PATH)
    jobs = pd.read_parquet(CHROMA_JOBS_PATH)
    _assert_files_exist(jobs, exp_root)
    man = pd.read_parquet(MANIFEST_PATH)[["stimulus_id", "factors"]]
    wf = jobs.merge(man, on="stimulus_id", how="left")
    ho = _held_out_mask(wf).values
    train = jobs[(~ho) & (jobs.image_condition == "image")]
    held = jobs[ho]  # all 4 conditions, eval
    return train.reset_index(drop=True), held.reset_index(drop=True)


def tag(seed):
    return f"qwen25omni-chroma-s{seed}"


def train(seed, smoke, exp_root):
    import torch
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments
    torch.manual_seed(seed)
    train_rows, held = _split(exp_root)
    print(f"[chroma] seed={seed}: {len(train_rows)} image-rows (modality-dropout "
          f"{MODE_P}) / {len(held)} held-out, tasks={CHROMA_TASKS}")
    model, processor, lm_path = load_qwen_omni_for_training()
    model.thinker = get_peft_model(model.thinker, build_lora_config(lm_path))
    assert_lora_applied(model.thinker, f"chroma-s{seed}")
    model.thinker.train()
    ds = DropoutDataset(train_rows.head(8) if smoke else train_rows, processor, exp_root, seed)
    out_dir = GPU_DIR / "track_g_chroma_ckpt" / tag(seed)
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
        print("[chroma] smoke done — inspect loss, then rerun without --smoke-test")
        return None, None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    model.thinker.save_pretrained(str(out_dir))
    print(f"[chroma] adapter saved to {out_dir}")
    return model, processor, held


def load_for_eval(seed):
    from peft import PeftModel
    base, processor, _ = load_qwen_omni_for_training()
    base.thinker = PeftModel.from_pretrained(
        base.thinker, str(GPU_DIR / "track_g_chroma_ckpt" / tag(seed)))
    _, held = _split(EXP_ROOT)
    return base, processor, held


def evaluate(seed, model, processor, held, exp_root):
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
                        "model": tag(seed), "seed": seed, "raw_response": raw,
                        "error": None, "ts": time.time()})
    out_path = RESULTS_DIR / f"responses__{tag(seed)}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"[chroma] seed={seed}: {len(results)} responses -> {out_path}")


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
