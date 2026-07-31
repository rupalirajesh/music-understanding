#!/usr/bin/env bash
# One-shot H100 runbook for what's left after Tracks C-F (PROJECT_STATE.md
# "Next actions" #4/#5/#11, updated 2026-07-31).
#
#   bash scripts/14_run_remaining_2.sh
#
# Only ONE step here is a ready-to-run command (Track F aug rerun, #11) --
# the other two open items (#4 L1 DSP floor, #5 harness validation) need new
# code / a manual decision each and are documented below, not scripted, so
# this runbook doesn't silently skip them. Read the comments before running.

cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/runlogs
LOG="results/runlogs/run_remaining2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "===== $* ====="; }
try()  { "$@" || echo "!! step failed (continuing): $*"; }

# ---------------------------------------------------------------------------
# #11 — Track F aug, clean rerun (leakage bug fixed 2026-07-31)
# ---------------------------------------------------------------------------
# generate_aug.py was sampling training pitches from the same 52-76 base_midi
# range as the frozen battery, which overlaps the held-out eval band
# (base_midi >= ~71.13 -- see train_track_c.py::_held_out_mask). Fixed to cap
# training pitches at MAX_TRAIN_MIDI=70 (1-semitone margin below the
# threshold). manifests/aug_train_jobs.parquet's METADATA (stimulus_id/
# ground_truth/prompt) happens to be byte-identical before/after the fix --
# truth depends on delta/detune, not base_midi -- so the fix only changes the
# actual WAV audio (frequencies), and stimuli/ is gitignored (never pushed).
# REGENERATE ON THIS BOX -- don't trust whatever WAVs already sit in
# stimuli/cents|tuning/ from the earlier leaky run:
step "Preflight: peft installed"
$PY -c "import peft" 2>/dev/null || $PY -m pip install -q -U peft

step "Regenerate aug stimuli WAVs + pitch-feature cache on THIS box (deterministic, same default seed=7, but the actual audio files aren't in git so they must be (re)built here)"
try $PY scripts/generate_aug.py --cents-per-cell 140 --tuning-per-cell 160
try $PY -m musicprobe.pitch_feats --manifest manifests/aug_train_jobs.parquet --out manifests/aug_pitch_feats.npz

step "Track F aug rerun: 3 seeds, corrected (leak-free) training data"
for S in 0 1 2; do
  try $PY gpu/train_track_f_pitchfuse.py --aug --seed "$S"
done

step "Track F aug: analyze (compare against the original leaky run in trackd_pitchfuseaug_summary.csv)"
try $PY gpu/analyze_track_d.py --tag pitchfuseaug
echo "Expect: fusion-null verdict (cents/tuning delta ~ns) should replicate."
echo "Watch: does the audio-only baseline's jump (cents 0.62->0.89 in the leaky"
echo "run) shrink now that training pitches can't leak into the held-out band?"
echo "If it shrinks a lot, the original 'more data helps audio-only' side-note"
echo "in commit 0aed136 was mostly leakage, not real generalization."

# ---------------------------------------------------------------------------
# #4 — L1 DSP floor: NOT scripted here, needs real implementation work
# ---------------------------------------------------------------------------
# Scoped precisely 2026-07-31 (PROJECT_STATE.md item 4): musicprobe/l1_baselines.py
# only covers 4/13 tasks (pitch_note_id, cents_discrimination, tempo_bpm,
# key_id). beats_per_bar, note_count, octave_id, tuning_judgment,
# instrument_id, interval_id, chord_quality, mode_id, progression_id have NO
# L1 baseline at all yet -- highest priority is beats_per_bar (flagged in
# item #2/PAPER.md as a low-confidence anomaly; an L1 beat-tracking floor
# would show whether the ground-truth labels are even DSP-recoverable, which
# would settle whether the wrong-audio anomaly is a task issue or a label
# issue). `pip install essentia` FAILS on the laptop (no wheel for this
# Python/platform, source build errors) -- confirmed 2026-07-31, do this here
# on the H100/Linux box where essentia actually installs. Suggested tools:
# essentia.standard.RhythmExtractor2013 (beat_per_bar/tempo_bpm), a proper
# key extractor (essentia.standard.KeyExtractor, replaces the naive-Krumhansl
# musicprobe.l1_baselines.key_estimate), and either essentia chord detection
# or a simple template-match for chord_quality/progression_id. Not attempted
# here blind (9 tasks, no local way to verify correctness) -- iterate on the
# box instead of guessing from the laptop.
step "L1 DSP floor -- SKIPPED, see comment block above. Needs new essentia-based code on this box."

# ---------------------------------------------------------------------------
# #5 — Qwen2-Audio harness validation: benchmark chosen, number not sourced
# ---------------------------------------------------------------------------
# Decided 2026-07-31: validate against MuChoMusic (ISMIR'24, arxiv 2408.01337,
# 1.1K MCQs), not MMAU-music -- it's a single well-defined public set (vs
# MMAU-music being a subscore inside the larger multi-domain MMAU benchmark),
# and it's the benchmark this project's own no-audio/wrong-audio control
# design is modeled after (RESEARCH_PLAN.md S:0.6), so it's the more
# meaningful fidelity check. Do NOT trust a secondhand number for this --
# pull Qwen2-Audio's exact published MuChoMusic accuracy directly from the
# MuChoMusic paper's model table (arxiv 2408.01337) or the Qwen2-Audio
# technical report (arxiv 2407.10759) before deciding what "replicated"
# means (exact eval subset, MCQ format, prompt wording all matter for an
# apples-to-apples check).
step "Qwen2-Audio harness validation -- SKIPPED, see comment block above. Pull the exact published number first, then build the eval subset."

# ---------------------------------------------------------------------------
# Generation battery: meter/mode manual listening (deliberately unscored)
# ---------------------------------------------------------------------------
# genmodel/score_gen.py intentionally does NOT auto-score meter/mode (needs
# downbeat tracking or a human ear, by design -- see PAPER.md Results,
# Generation models section). WAVs are gitignored. Pull them from this box
# and listen; nothing to run here.
step "Generation battery meter/mode -- manual listening pass, not scriptable. See genmodel/outputs/ on this box."

# ---------------------------------------------------------------------------
step "Commit + push results"
git add results/ manifests/aug_train_jobs.parquet manifests/aug_pitch_feats.npz
git commit -m "Track F aug rerun on leak-free training data" \
  && git push || echo "nothing to commit or push failed — check manually"
step "DONE — check the log above for '!! step failed' lines"
