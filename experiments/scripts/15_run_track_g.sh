#!/usr/bin/env bash
# One-shot runbook for Track G (novel, 2026-07-31): does a chromagram image
# help the harmonic task cluster (key_id/mode_id/chord_quality/interval_id)
# that Tracks C-F never targeted?
#
#   bash scripts/15_run_track_g.sh
#
# Steps 1-2 (chromagram render + chroma-jobs build) need NO GPU and were
# already run + verified locally 2026-07-31 (664 PNGs rendered across the
# whole battery for a valid wrong-image draw pool, 1760-row chroma_jobs.parquet
# built, held-out split sanity-checked: non-empty train/held for all 4 tasks)
# -- idempotent, rerunning here just confirms nothing's missing before the
# GPU steps. Steps 3-4 need the H100 box and are UNVERIFIED on hardware --
# smoke-test first, same discipline as scripts/13_run_track_d.sh.
#
# Design (same rigor as Track D's conclusive/force runs, applied from the
# start this time -- no OOD single-arm first pass): modality-dropout training
# (audio+chromagram / chromagram-only / audio-only, so both eval conditions
# stay in-distribution) + wrong_image / image_wrong_audio controls + held-out
# SOUNDFONT split (these 4 tasks all have a soundfont factor, verified) +
# paired McNemar eval over 3 seeds.
#
# Prereqs on the box (one-time; same as Track D/E):
#   pip install -r requirements.txt torch accelerate soundfile pandas pyarrow
#   librosa matplotlib openpyxl scikit-learn torchaudio peft

cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/runlogs
LOG="results/runlogs/run_trackg_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "===== $* ====="; }
try()  { "$@" || echo "!! step failed (continuing): $*"; }

step "Preflight: peft installed"
$PY -c "import peft" 2>/dev/null || $PY -m pip install -q -U peft

# ---------- Part 1: groundwork (CPU, idempotent, already run+committed) -----
step "Render chromagrams (whole battery -- needed for a valid wrong-image draw pool; skips existing PNGs)"
$PY scripts/render_chromagrams.py --tasks

step "Build the chroma-jobs hygiene layer (image/no_image/wrong_image/image_wrong_audio)"
$PY -m musicprobe.image_jobs --kind chromagram

# ---------- Part 2: Track G — modality-dropout LoRA on Qwen2.5-Omni-7B -----
step "Track G: smoke test (8 steps, 8 examples, no checkpoint saved)"
$PY gpu/train_track_g_chroma.py --seed 0 --smoke-test
# ^ NOT wrapped in try(): if this fails, STOP and read the assertion message.

step "Track G: full training + held-out eval, 3 seeds"
for S in 0 1 2; do
  try $PY gpu/train_track_g_chroma.py --seed "$S"
done

step "Track G: analyze (results/trackA/trackg_chroma_summary.csv + graph)"
try $PY gpu/analyze_track_g.py
echo "Read: does chromagram help (image vs no_image, McNemar)? Does the model"
echo "actually USE it (wrong_image vs no_image)? A help-but-doesn't-use pattern"
echo "would be surprising and worth a second look before trusting it."

# ---------- Ship it ----------------------------------------------------------
step "Commit + push results (gpu/track_g_chroma_ckpt/ is gitignored — only responses/summary go up)"
git add results/ manifests/chroma_jobs.parquet
git commit -m "Track G: chromagram front-end for key_id/mode_id/chord_quality/interval_id" \
  && git push || echo "nothing to commit or push failed — check manually"
step "DONE — check the log above for '!! step failed' lines"
