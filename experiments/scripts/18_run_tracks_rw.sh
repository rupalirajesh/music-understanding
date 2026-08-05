#!/usr/bin/env bash
# READ scripts/RUNBOOK_tracks_lq_rw.md FIRST -- it has the full context,
# rationale per track, judgment-call guidance for the checkpoints below,
# known limitations, and the leakage rule that must never be broken if you
# touch this code. This script is just the command sequence; that doc is
# the why.
#
# One-shot runbook for Tracks R-W (2026-08-05): the rhythm-cluster
# representation-ladder sequence -- tempogram (R), peak-picked tempogram (S),
# onset-strength line graph (T), zoomed onset-strength line graph (U),
# beat/onset grid "rhythm-roll" (V), rhythm necklace / circular polygon (W).
# First causal fine-tuning of ANY kind on tempo_bpm/beats_per_bar -- this
# cluster has had zero prior Track C-H style intervention.
#
#   bash scripts/18_run_tracks_rw.sh
#
# Policy (Rupali's call, 2026-08-05): run the FULL sequence, not stop-early.
# See PROJECT_STATE.md next action 14 for the full design writeup.
#
# Leakage note (read before trusting Track V/W's images): the metrical
# grid (V) and circle circumference (W) are sized from a DETECTED
# periodicity (scripts/render_rhythm_repr.py:_detect_click_period, median
# inter-onset interval from librosa's own onset detector), never from the
# ground-truth beats-per-bar label -- verified against 15 real stimuli
# spanning all 5 beats-per-bar categories (3/4/5/6/7), 12/15 correct cycle-
# length detection. Not perfect (this is a genuinely hard MIR problem, same
# reason PROJECT_STATE flags proper meter detection as needing essentia on
# this box, not laptop heuristics) but real and non-leaking.
#
# Steps 1-2 need NO GPU and were already run + verified locally 2026-08-05
# (all 6 representations rendered across the WHOLE battery for a valid
# wrong-image draw pool). Idempotent. Steps 3-4 need the H100 box and are
# UNVERIFIED on hardware -- smoke-test first, same discipline as every gpu/
# script in this project.
#
# Prereqs on the box (one-time; same as Track G/H/L-Q):
#   pip install -r requirements.txt torch accelerate soundfile pandas pyarrow
#   librosa matplotlib openpyxl scikit-learn torchaudio peft

cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/runlogs
LOG="results/runlogs/run_tracks_rw_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "===== $* ====="; }
try()  { "$@" || echo "!! step failed (continuing): $*"; }

step "Preflight: peft installed"
$PY -c "import peft" 2>/dev/null || $PY -m pip install -q -U peft

# ---------- Part 1: groundwork (CPU, idempotent, already run+committed) -----
step "Render all 6 rhythm representations (whole battery -- needed for a valid wrong-image draw pool; skips existing PNGs)"
$PY scripts/render_rhythm_repr.py --kind all

# ---------- Part 2: Tracks R-W — modality-dropout LoRA on Qwen2.5-Omni-7B --
for TRACK in R S T U V W; do
  step "Track $TRACK: smoke test (8 steps, 8 examples, no checkpoint saved)"
  $PY gpu/train_track_repr.py --track "$TRACK" --seed 0 --smoke-test

  step "Track $TRACK: full training + held-out eval, 3 seeds"
  for S in 0 1 2; do
    try $PY gpu/train_track_repr.py --track "$TRACK" --seed "$S"
  done

  step "Track $TRACK: analyze"
  try $PY gpu/analyze_track_repr.py --track "$TRACK"
done

step "Cross-track comparison (which representation wins on which task)"
try $PY gpu/analyze_track_repr.py --track R --compare S T U V W

# ---------- Ship it ----------------------------------------------------------
step "Commit + push results (gpu/track_*_ckpt/ is gitignored -- only responses/summary go up)"
git add results/ manifests/*_jobs.parquet
git commit -m "Tracks R-W: rhythm-cluster representation ladder (tempogram -> rhythm necklace)" \
  && git push || echo "nothing to commit or push failed — check manually"
step "DONE — check the log above for '!! step failed' lines"
