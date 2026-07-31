#!/usr/bin/env bash
# One-shot runbook for Track H (novel, 2026-07-31): does an in-AUDIO
# reference tone fix absolute tuning (tuning_judgment), the way Track D-zoom's
# rendered reference-line image did — without switching modality at all?
#
#   bash scripts/16_run_track_h.sh
#
# Steps 1-2 (reftone WAV render + reftone-jobs build) need NO GPU and were
# already run + verified locally 2026-07-31 (120 stimuli x 2 new variants,
# 360-row reftone_jobs.parquet, held-out split sanity-checked: 93 train / 27
# held-out stimuli, no train/held stimulus overlap) -- idempotent, rerunning
# here just confirms nothing's missing before the GPU steps. Steps 3-4 need
# the H100 box and are UNVERIFIED on hardware — smoke-test first, same
# discipline as scripts/13_run_track_d.sh / 15_run_track_g.sh.
#
# Design: dropout-style training (plain / reftone mixed per example, NEVER
# wrong_reftone — same convention as wrong_image/wrong_audio elsewhere) so
# both eval conditions stay in-distribution + held-out base_midi-quintile
# split (tuning_judgment has no soundfont factor, train_track_c's fallback
# applies) + paired McNemar eval over 3 seeds + a wrong_reftone mechanism
# control: a genuine target-vs-reference comparison should be MISLED by a
# wrong reference; a model just reacting to "two tones present" won't be.
#
# Prereqs on the box (one-time; same as Track D/E/G):
#   pip install -r requirements.txt torch accelerate soundfile pandas pyarrow
#   librosa matplotlib openpyxl scikit-learn torchaudio peft

cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/runlogs
LOG="results/runlogs/run_trackh_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "===== $* ====="; }
try()  { "$@" || echo "!! step failed (continuing): $*"; }

step "Preflight: peft installed"
$PY -c "import peft" 2>/dev/null || $PY -m pip install -q -U peft

# ---------- Part 1: groundwork (CPU, idempotent, already run+committed) -----
step "Render reftone/wrong_reftone WAVs (pure numpy synthesis, skips existing files)"
$PY scripts/render_reftones.py

step "Build the reftone-jobs hygiene layer (plain/reftone/wrong_reftone)"
$PY -m musicprobe.reftone_jobs

# ---------- Part 2: Track H — dropout-trained LoRA on Qwen2.5-Omni-7B ------
step "Track H: smoke test (8 steps, 8 examples, no checkpoint saved)"
$PY gpu/train_track_h_reftone.py --seed 0 --smoke-test
# ^ NOT wrapped in try(): if this fails, STOP and read the assertion message.

step "Track H: full training + held-out eval, 3 seeds"
for S in 0 1 2; do
  try $PY gpu/train_track_h_reftone.py --seed "$S"
done

step "Track H: analyze (results/trackA/trackh_reftone_summary.csv + graph)"
try $PY gpu/analyze_track_h.py
echo "Read: does the reference tone help (reftone vs plain, McNemar)? Does a"
echo "WRONG reference mislead the model (wrong_reftone vs plain)? If reftone"
echo "helps but wrong_reftone does NOT hurt relative to reftone, the model may"
echo "be reacting to 'audio got longer/more complex' rather than genuinely"
echo "comparing target-to-reference — read the raw responses before trusting"
echo "a positive result blindly, same discipline as every other track here."

# ---------- Ship it ----------------------------------------------------------
step "Commit + push results (gpu/track_h_reftone_ckpt/ is gitignored — only responses/summary go up)"
git add results/ manifests/reftone_jobs.parquet
git commit -m "Track H: in-audio reference tone for tuning_judgment" \
  && git push || echo "nothing to commit or push failed — check manually"
step "DONE — check the log above for '!! step failed' lines"
