"""Evaluate the ALREADY-TRAINED Track D-zoom checkpoint against real-timbre
audio (PROJECT_STATE.md next action 24) -- this is inference on an existing
adapter, NOT a new training run. Same checkpoint, same model, same
image-vs-audio conditions as the original synthetic-battery run; the only
thing that changes is which `held` jobs table gets evaluated.

Checkpoint location (confirmed by reading gpu/train_track_d_force.py --
Track D-zoom is that script run with --image-kind f0zoom, not a separate
file): gpu/track_d_force_ckpt/qwen25omni-zoom-s{seed}/. If that directory
doesn't exist, the checkpoint was cleared from this box since the original
2026-07-29 run and needs retraining first (rerun
train_track_d_force.py --seed {seed} --image-kind f0zoom) --
this script does NOT retrain, it only evaluates.

Prereq -- CORRECTED 2026-08-21, the note below was wrong: only
manifests/real_nsynth_dzoom_jobs.parquet is actually committed (verified via
`git ls-files`). The rendered stimuli/f0zoom/real_nsynth/*.png images and the
underlying real_nsynth/*.wav audio are NOT committed (same gitignore pattern
that excludes all of stimuli/) -- they must be regenerated on whichever box
runs this eval, via:
  python -m musicprobe.real_music_nsynth --n 60 --seed 0        # pulls NSynth
                                                                 # audio from HF
  python -m musicprobe.real_music_nsynth --dzoom-jobs --seed 0  # renders images
before running this script. Both are deterministic/seeded, so re-running them
on a new box reproduces the same content the manifest was built against.

  python gpu/eval_track_dzoom_real.py --seed 0
  python gpu/eval_track_dzoom_real.py --seed 0 --limit 20   # smoke test first

Writes results/trackA/responses__qwen25omni-zoom-real_nsynth-s{seed}.parquet,
SAME schema as every other track's responses__*.parquet (job_id, stimulus_id,
task, image_condition, model, seed, raw_response, error, ts) -- reuse
gpu/analyze_track_d.py's scoring path against it, or musicprobe.scoring
directly, same as any other track's output.

UNVERIFIED on hardware (no GPU on the laptop, same status as every other
gpu/ script before its first real run) -- the checkpoint-loading and
generation loop are copied verbatim from train_track_d_force.py's already-
working load_for_eval()/evaluate() (same PeftModel.from_pretrained call,
same content-building, same generate() call), only the jobs source changed,
so the UNVERIFIED surface here is much smaller than a from-scratch script.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
from train_track_d import load_qwen_omni_for_training  # noqa: E402
from musicprobe.config import EXP_ROOT, RESULTS_DIR, MANIFEST_DIR  # noqa: E402

CKPT_DIR = GPU_DIR / "track_d_force_ckpt"
JOBS_PATH = MANIFEST_DIR / "real_nsynth_dzoom_jobs.parquet"


def tag(seed, no_lora=False):
    return f"qwen25omni-{'base' if no_lora else 'zoom'}-real_nsynth-s{seed}"


def load_checkpoint(seed, no_lora=False):
    """no_lora=True loads the plain base model (no adapter) against the SAME
    real-NSynth jobs table -- added 2026-08-19 so a real baseline-vs-fine-tuned
    delta can be computed on identical real-timbre stimuli, rather than only
    comparing this checkpoint's real-audio accuracy against the ORIGINAL
    synthetic-battery numbers (a different confound: timbre and checkpoint both
    differ there, not just checkpoint). See BENCHMARK_LANDSCAPE.md Sec6 /
    PROJECT_STATE.md next action 26."""
    base, processor, _ = load_qwen_omni_for_training()
    if no_lora:
        return base, processor
    from peft import PeftModel
    ckpt_path = CKPT_DIR / f"qwen25omni-zoom-s{seed}"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found -- the D-zoom checkpoint isn't on this box (cleared, or "
            "never saved from this seed). Retrain first: python gpu/train_track_d_force.py "
            f"--seed {seed} --image-kind f0zoom (then rerun this script -- do NOT skip "
            "straight to evaluating, an untrained/base model will look like a null result "
            "that isn't real).")
    base.thinker = PeftModel.from_pretrained(base.thinker, str(ckpt_path))
    return base, processor


def evaluate(seed, model, processor, jobs, exp_root, limit=None, no_lora=False):
    import torch
    model.eval()
    if limit:
        # NOT groupby(...).apply(lambda g: g.head(n)) -- confirmed 2026-08-21 on
        # pandas 3.0.5: groupby().apply() now drops the grouping column from what
        # the lambda receives by default, silently deleting image_condition from
        # the result. GroupBy.head() is a different code path and doesn't have
        # this problem.
        jobs = jobs.groupby("image_condition", group_keys=False).head(
            max(1, limit // jobs.image_condition.nunique()))
    results = []
    for n, row in enumerate(jobs.itertuples(), 1):
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
                        "model": tag(seed, no_lora), "seed": seed, "raw_response": raw,
                        "error": None, "ts": time.time()})
        if n % 50 == 0:
            print(f"  {n}/{len(jobs)}")
    out_path = RESULTS_DIR / f"responses__{tag(seed, no_lora)}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"[eval-dzoom-real] seed={seed} no_lora={no_lora}: {len(results)} responses -> {out_path}")
    if no_lora:
        print("[eval-dzoom-real] this is the BASELINE half -- run without --no-lora too (same "
              "seed, same jobs) and diff acc by (task, image_condition) against this file for "
              "the real controlled delta. Don't compare this alone against "
              "trackd_zoom_summary.csv's synthetic-battery numbers -- that's a different "
              "stimulus set, not a controlled comparison.")
    else:
        print("[eval-dzoom-real] this is the FINE-TUNED half -- also run with --no-lora (same "
              "seed, same jobs) for the real baseline-vs-fine-tuned delta on real timbre. "
              "trackd_zoom_summary.csv's acc_audio/acc_image (cents 0.55->0.94, tuning "
              "0.53->0.89) are the ORIGINAL synthetic-battery numbers for reference only -- "
              "not a substitute for the --no-lora run on these same real-NSynth jobs.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total jobs, roughly balanced across the 4 image_conditions -- "
                         "use this for a smoke test before the full 1440-job run")
    ap.add_argument("--exp-root", default=str(EXP_ROOT))
    ap.add_argument("--no-lora", action="store_true",
                    help="run the plain base model (no D-zoom adapter) against these same "
                         "real-NSynth jobs -- run both with and without this flag (same --seed) "
                         "to get the real controlled baseline-vs-fine-tuned delta on real timbre")
    a = ap.parse_args()
    if not JOBS_PATH.exists():
        raise SystemExit(f"{JOBS_PATH} missing -- run "
                         "`python -m musicprobe.real_music_nsynth --dzoom-jobs` first "
                         "(laptop, CPU-only, already committed if you pulled latest).")
    jobs = pd.read_parquet(JOBS_PATH)
    model, processor = load_checkpoint(a.seed, no_lora=a.no_lora)
    evaluate(a.seed, model, processor, jobs, Path(a.exp_root), limit=a.limit, no_lora=a.no_lora)
