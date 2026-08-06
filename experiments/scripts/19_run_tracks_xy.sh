#!/usr/bin/env bash
# READ scripts/RUNBOOK_tracks_xy.md FIRST -- it has the full context, why
# these two tracks exist, and the same leakage/judgment-call guidance as
# scripts/RUNBOOK_tracks_lq_rw.md. This script is just the command sequence.
#
# One-shot runbook for Tracks X/Y (2026-08-06): the missing zoom+explicit-
# reference combination that Tracks L-Q/R-W never tested. L-Q/R-W tested
# zoom (M/O/U) and an explicit reference/richness ingredient (P/V)
# separately, but Track D-zoom only fixed pitch by combining BOTH at once
# (zoom + reference line together) -- neither ingredient alone worked there
# either (Track D force = reference without zoom, null; Track D conclusive's
# precursor = resolution without reference, null). This is that combination,
# tried once on each cluster:
#   X -- zoomed peak-picked chroma + estimated-tonic reference row/label
#        (harmony: key_id/mode_id/chord_quality/interval_id)
#   Y -- zoomed onset-strength rhythm-roll (onsets vs. detected pulse grid,
#        at Track U's finer time resolution) (rhythm: tempo_bpm/beats_per_bar)
#
#   bash scripts/19_run_tracks_xy.sh
#
# Steps 1-2 (render + job-manifest build, held-out split) need NO GPU and
# were already run + verified locally 2026-08-06 (both representations
# rendered across the WHOLE battery -- 1248/1248, 0 errors -- and both
# held-out splits verified to exactly match their parent ladder's split:
# Track X = 287 train/612 held, same as Track G/L-Q; Track Y = 127 train/
# 132 held, same as Track R-W; zero train/held overlap in either). Steps
# 3-4 need the H100 box and are UNVERIFIED on hardware -- smoke-test first,
# same discipline as every gpu/ script in this project.
#
# Prereqs on the box (one-time; same as Track G/H/L-Q/R-W):
#   pip install -r requirements.txt torch accelerate soundfile pandas pyarrow
#   librosa matplotlib openpyxl scikit-learn torchaudio peft

cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/runlogs
LOG="results/runlogs/run_tracks_xy_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "===== $* ====="; }
try()  { "$@" || echo "!! step failed (continuing): $*"; }

step "Preflight: peft installed"
$PY -c "import peft" 2>/dev/null || $PY -m pip install -q -U peft

# ---------- Part 1: groundwork (CPU, idempotent, already run+committed) -----
step "Render Track X (chroma_zoom_ref, whole battery) -- skips existing PNGs"
$PY scripts/render_harmony_repr.py --kind chroma_zoom_ref --scope battery
step "Render Track Y (rhythm_roll_zoom, whole battery) -- skips existing PNGs"
$PY scripts/render_rhythm_repr.py --kind rhythm_roll_zoom --scope battery

# job manifests are built lazily on first use by gpu/image_track_common.py's
# split(), same as every earlier track -- no separate build step needed.

# ---------- Part 2: Tracks X/Y — modality-dropout LoRA on Qwen2.5-Omni-7B --
for TRACK in X Y; do
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

step "Track X vs. its ladder (does zoom+reference beat every L-Q variant tried alone?)"
try $PY gpu/analyze_track_repr.py --track X --compare L M N O P Q
step "Track Y vs. its ladder (does zoom+reference beat every R-W variant tried alone?)"
try $PY gpu/analyze_track_repr.py --track Y --compare R S T U V W

# ---------- Ship it ----------------------------------------------------------
step "Commit + push results (stimuli/ and gpu/track_*_ckpt/ are gitignored -- only responses/summary/manifests go up, same convention as every earlier track)"
git add results/ manifests/chroma_zoom_ref_jobs.parquet manifests/rhythm_roll_zoom_jobs.parquet
git commit -m "Tracks X/Y: zoom+explicit-reference combination for harmony + rhythm" \
  && git push || echo "nothing to commit or push failed — check manually"
step "DONE — check the log above for '!! step failed' lines"
