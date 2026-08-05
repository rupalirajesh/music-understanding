#!/usr/bin/env bash
# READ scripts/RUNBOOK_tracks_lq_rw.md FIRST -- it has the full context,
# rationale per track, judgment-call guidance for the checkpoints below,
# known limitations, and the leakage rule that must never be broken if you
# touch this code. This script is just the command sequence; that doc is
# the why.
#
# One-shot runbook for Tracks L-Q (2026-08-05): the harmony-cluster
# representation-ladder sequence -- peak-picked chroma (L), zoomed
# peak-picked chroma (M), multi-pitch line graph (N), zoomed line graph (O),
# piano-roll (P), tonal centroid / Tonnetz (Q). Same task cluster as Track G
# (key_id/mode_id/chord_quality/interval_id), each testing a different
# representation of the same audio.
#
#   bash scripts/17_run_tracks_lq.sh
#
# Policy (Rupali's call, 2026-08-05): run the FULL sequence, not stop-early --
# the goal is comparing which representation works best, not just finding
# one fix. See PROJECT_STATE.md next action 13 for the full design writeup
# and gpu/image_track_common.py for why this is registry-based (one shared
# harness script for all six tracks) instead of six near-duplicate files
# like Track G/H used.
#
# Steps 1-2 (render + job-manifest build) need NO GPU and were already run +
# verified locally 2026-08-05 (all 6 representations rendered across the
# WHOLE battery -- not just the 4 harmony tasks -- for a valid wrong-image
# draw pool, same reasoning as Track G's 664-PNG render). Idempotent,
# rerunning here just confirms nothing's missing before the GPU steps.
# Steps 3-4 need the H100 box and are UNVERIFIED on hardware -- smoke-test
# first, same discipline as every gpu/ script in this project.
#
# Prereqs on the box (one-time; same as Track G/H):
#   pip install -r requirements.txt torch accelerate soundfile pandas pyarrow
#   librosa matplotlib openpyxl scikit-learn torchaudio peft

cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/runlogs
LOG="results/runlogs/run_tracks_lq_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "===== $* ====="; }
try()  { "$@" || echo "!! step failed (continuing): $*"; }

step "Preflight: peft installed"
$PY -c "import peft" 2>/dev/null || $PY -m pip install -q -U peft

# ---------- Part 1: groundwork (CPU, idempotent, already run+committed) -----
step "Render all 6 harmony representations (whole battery -- needed for a valid wrong-image draw pool; skips existing PNGs)"
$PY scripts/render_harmony_repr.py --kind all

# job manifests are built lazily on first use by gpu/image_track_common.py's
# split(), same as Track G -- no separate build step needed here.

# ---------- Part 2: Tracks L-Q — modality-dropout LoRA on Qwen2.5-Omni-7B --
for TRACK in L M N O P Q; do
  step "Track $TRACK: smoke test (8 steps, 8 examples, no checkpoint saved)"
  $PY gpu/train_track_repr.py --track "$TRACK" --seed 0 --smoke-test
  # ^ NOT wrapped in try(): if this fails, STOP and read the assertion message
  #   before spending real GPU time on this track.

  step "Track $TRACK: full training + held-out eval, 3 seeds"
  for S in 0 1 2; do
    try $PY gpu/train_track_repr.py --track "$TRACK" --seed "$S"
  done

  step "Track $TRACK: analyze"
  try $PY gpu/analyze_track_repr.py --track "$TRACK"
done

step "Cross-track comparison (which representation wins on which task)"
try $PY gpu/analyze_track_repr.py --track L --compare M N O P Q

# ---------- Ship it ----------------------------------------------------------
step "Commit + push results (gpu/track_*_ckpt/ is gitignored -- only responses/summary go up)"
git add results/ manifests/*_jobs.parquet
git commit -m "Tracks L-Q: harmony-cluster representation ladder (peak-picked chroma -> tonal centroid)" \
  && git push || echo "nothing to commit or push failed — check manually"
step "DONE — check the log above for '!! step failed' lines"
