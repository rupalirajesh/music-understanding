#!/usr/bin/env bash
# One-shot runbook for Track D Phase 1 (RESEARCH_PLAN.md §12.2): does a
# spectrogram image, added alongside audio, change accuracy on the
# alignment-fixable shortlist -- with a wrong-image control so a positive
# result can't just mean "the model learned to read the image and ignore
# audio."
#
#   bash scripts/13_run_track_d.sh
#
# Steps 1-2 (spectrogram render + image-jobs build) need NO GPU and were
# already run + verified locally 2026-07-24/25 -- they're idempotent
# (render skips existing PNGs, image_jobs rebuild is deterministic) so
# rerunning here is cheap and just confirms nothing's missing before the
# GPU steps. Steps 3-4 need the H100 box and are UNVERIFIED on hardware —
# smoke-test first, same discipline as scripts/11_run_track_c.sh.

cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/runlogs
LOG="results/runlogs/run_trackd_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "===== $* ====="; }
try()  { "$@" || echo "!! step failed (continuing): $*"; }

step "Preflight: peft installed"
$PY -c "import peft" 2>/dev/null || $PY -m pip install -q -U peft

# ---------- Part 1: groundwork (CPU, idempotent) ----------------------------
step "Render spectrograms (skips PNGs that already exist)"
$PY scripts/12_render_spectrograms.py

step "Build the image hygiene layer (image/no_image/wrong_image jobs)"
$PY -m musicprobe.image_jobs

# ---------- Part 2: Track D — single-arm LoRA on Qwen2.5-Omni-7B -----------
step "Track D: smoke test (8 steps, 8 examples, no checkpoint saved)"
$PY gpu/train_track_d.py --smoke-test
# ^ NOT wrapped in try(): if this fails, STOP and read the assertion message
# — it will name exactly which submodule didn't resolve, per the script's
# docstring (the Thinker language-model path is the most likely culprit).

step "Track D: full training + held-out eval across image/no_image/wrong_image"
try $PY gpu/train_track_d.py

step "Track D: accuracy by task x image_condition (results/trackA/trackd_image_summary.csv)"
echo "printed above by train_track_d.py itself — read it before concluding anything:"
echo "  wrong_image accuracy close to image accuracy => model is ignoring the image"
echo "  (same 'is it actually using this input' logic as the wrong_audio control)"

# ---------- Ship it ----------------------------------------------------------
step "Commit + push results (gpu/track_d_checkpoints/ is gitignored — only responses/summary go up)"
git add results/ experiments/results/ manifests/image_jobs.parquet 2>/dev/null
git commit -m "Track D Phase 1: Qwen2.5-Omni-7B + spectrogram-image LoRA run" \
  && git push || echo "nothing to commit or push failed — check manually"
step "DONE — check the log above for '!! step failed' lines"
