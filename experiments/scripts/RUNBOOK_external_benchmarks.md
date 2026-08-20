# Runbook: external benchmark suite (CMI-Bench, PitchBench, BASS) — for whoever runs this on the H100 box

Written 2026-08-19. Companion to `BENCHMARK_LANDSCAPE.md` §6 and `PROJECT_STATE.md`
next action 26 — read §6 first for the why. Short version: the mentor's framing is
that a benchmark-suite eval only means something if Qwen2.5-Omni-7B (the Track C-Z
LoRA base) does *poorly* across the board first. MUSE Benchmark already answered
this for free (their own logs, parsed — `gpu/parse_musebench_qwen.py`, no GPU
needed, already run): at/near chance on all 5 advanced tasks. This runbook covers
the three benchmarks that need an actual inference run because no existing
Qwen2.5-Omni log exists to mine.

## 0. Priority order

1. **CMI-Bench** — the portfolio's NL-format anchor (§5). Real audio, real MIR
   metrics, covers key/melody/beat/vocal-technique/instrument-technique in one
   place, includes VocalSet as one of its own sub-tasks (no separate VocalSet run
   needed). Fully built and schema-verified.
2. **PitchBench** — closest published neighbor to this project's own battery, good
   sanity baseline. Fully built, schema-verified via HF's datasets-server API
   directly (not the paper text).
3. **BASS** — supplementary (real-audio scale, not a theory-task source — its
   "musicological analysis" category is gene/attribute dominance, not harmony).
   **Blocked on audio resolution**, not fully runnable yet — see §3.

Not built: **GuitarSet** (no existing chat-QA wrapper anywhere, incl. CMI-Bench —
would need a `real_music_*.py`-style wrapper built from scratch, same pattern as
`musicprobe/real_music_medleydb.py`; lower priority, technique-depth supplement
only). **MusICA-MetaBench** (no public code repo found as of 2026-08-19 — already
deprioritized in §5 as framework-only/MCQ-only).

## 1. CMI-Bench

```
git clone https://github.com/nicolaus625/CMI-bench.git <cmi_dir>
cd <cmi_dir> && wget https://huggingface.co/datasets/nicolaus625/CMI-bench/resolve/main/test_Data.zip
unzip test_Data.zip -d test

cd <path to this repo>/experiments
python gpu/eval_cmibench.py --cmi-bench-dir <cmi_dir> --limit 5    # smoke test, eyeball responses first
python gpu/eval_cmibench.py --cmi-bench-dir <cmi_dir>              # full run (PRIORITY_TASKS: GS_key,
                                                                     # Nsynth_pitch, VocalSet_tech, Guzheng_Tech,
                                                                     # gtzan_beat, ballroom_beat, MedleyDB)
cd <cmi_dir>
# evaluate.py's --model choices= allowlist doesn't include "qwen25omni" -- add it
# there (one line) before running, their scoring functions are keyed off the task
# name in the filename, not the model name, so this is safe
python evaluate.py --model qwen25omni --task all
```

Cross-reference while reading the output: `BENCHMARK_LANDSCAPE.md` §2 already has
Qwen2-Audio's CMI-Bench numbers (GTZAN 72.07%, GiantSteps key 8.28 vs SOTA 74.3,
beat F-measure 23.69 vs SOTA 88.3, melody-extraction 5.06% vs SOTA 72.3) — the
Qwen2.5-Omni run above is directly comparable to those, same benchmark same metrics.

Also worth noting for the real-music dataset hunt (`PROJECT_STATE.md` next action
23): CMI-Bench's `test_Data.zip` bundles real GiantSteps-key audio directly
(`data/GS-key/giantsteps_clips/wav/...`) — the original Beatport-CDN download path
for that dataset was confirmed dead in that earlier hunt, but this zip may be a
working mirror. Worth checking before writing off GiantSteps key entirely.

## 2. PitchBench

No separate download step — audio streams from HF directly.

```
pip install datasets soundfile librosa   # if not already present
python gpu/eval_pitchbench.py --config pitchbench_a1_single_pitch_id --limit 5   # smoke test
python gpu/eval_pitchbench.py                                                     # all 30 configs, resumable per-config
```

Only category A's exact 4-field schema (`gt_midi`/`prompt_midi` etc.) was verified
directly against the real dataset (`datasets-server` API, 2026-08-19). Categories
B-F (timestamps, chords, sequences, robustness) were NOT individually checked — the
script uses the same `prompt_midi`/`gt_midi` field names everywhere and will raise
a `KeyError` rather than silently mis-score any config whose fields differ. If that
happens, `--config <name> --limit 1` and print `ds.column_names` to see the real
shape before deciding how to extend the script for that config.

## 3. BASS — blocked, do this part first

`gpu/eval_bass.py --build-manifest --category <cat>` pulls the real dataset (no
audio needed) and confirms field names. The blocker: each row's `audio` field is a
bare filename (e.g. `"collab_analysis_2031.wav"`), not a resolvable path or embedded
bytes, and there's a separate `youtube_url` field alongside it. Where the actual
`.wav` files live wasn't found during this pass (checked the HF dataset page and
datasets-server API, no obvious audio distribution alongside the metadata parquet).
Two ways to unblock, in order of preference:
1. Check the BASS GitHub repo's issues/discussions or email the authors
   (oahia@cs.washington.edu / minjang@cs.washington.edu, per their README) — they
   may have a private/gated audio bundle analogous to CMI-Bench's `test_Data.zip`.
2. Fall back to `youtube_url` + `yt-dlp`, same fragile/lossy path this project
   already deprioritized once for MuChoMusic's MusicCaps subset
   (`gpu/eval_muchomusic.py`'s BLOCKER section) — same risk (clips get taken down,
   not reliably comparable to any published number) applies here.
Given BASS is supplementary not anchor, this is lower-priority than getting
CMI-Bench and PitchBench numbers in first.

## 3.5 Baseline vs fine-tuned — added 2026-08-19, read this before running anything twice

Rupali's ask: don't just check whether the baseline model does poorly (§0-3 above) — check
whether the Track C-Z fine-tuning actually **improves** performance, on real music, not just
the synthetic battery. This needs one clarification before running anything, because "run
checkpoint X against benchmark Y" means two different things depending on which checkpoint:

- **Track C (plain LoRA, audio-in/text-out, no special front-end)** — this is the ONLY
  Track C-Z checkpoint that takes the exact same input shape as CMI-Bench/PitchBench/BASS's
  own prompts (audio + text question, nothing else injected). Every script built today
  (`eval_cmibench.py`, `eval_pitchbench.py`, `eval_bass.py`) now takes `--lora-checkpoint` for
  exactly this: point it at Track C's saved adapter dir, same `.thinker`-only wrap used
  everywhere else in this project, and you get a real, apples-to-apples baseline-vs-fine-tuned
  delta on real/semi-real audio. **This is the test to run first** — cheapest, most direct
  answer to "does fine-tuning transfer to real music."
- **Track E (f0-as-text) and Track D-zoom (zoomed F0 image + reference line)** — these are
  NOT plug-and-play into the same harnesses. Their whole mechanism is a special front-end
  (extracted pitch numbers injected into the prompt text, or a rendered chart image alongside
  the audio) — wrapping their checkpoint with a PLAIN audio-only prompt tests something real
  but different: "does this checkpoint still work reasonably when you don't give it the input
  it was fine-tuned to expect," not "does the winning pitch fix generalize to real music."
  Testing the actual claim needs the SAME front-end built for the new benchmark's audio, not
  just the checkpoint swapped in. That pipeline already exists for one real-audio source —
  `eval_track_dzoom_real.py` (NSynth, real instrument timbre, F0-extraction + zoom-render
  reused unmodified from the synthetic pipeline) — and was extended today with `--no-lora` so
  it now produces a real controlled baseline-vs-fine-tuned delta on the SAME real-NSynth jobs
  (previously it only had the fine-tuned half; comparing that against the ORIGINAL
  synthetic-battery numbers wasn't a controlled comparison, since timbre AND checkpoint both
  differed). **This is ready to run right now, no new engineering**:
  ```
  python gpu/eval_track_dzoom_real.py --seed 0             # fine-tuned half
  python gpu/eval_track_dzoom_real.py --seed 0 --no-lora   # baseline half, same jobs
  ```
  MedleyDB's equivalent is still blocked on Zenodo access (PROJECT_STATE.md next action 23).
  Extending this same front-end pipeline to CMI-Bench's `Nsynth_pitch`/`GS_key` clips or
  PitchBench's audio (build a `build_dzoom_jobs`-equivalent for their stimuli, mirroring
  `musicprobe/real_music_nsynth.py`) is a real next step but not done here — flagging as scoped
  future work, not silently building it blind.

**Priority order for the actual "did fine-tuning help" question**: (1) `eval_track_dzoom_real.py`
both halves — already built, run it first, zero new code. (2) Track C's checkpoint through
CMI-Bench/PitchBench/BASS via `--lora-checkpoint` — real engineering already done today, just
needs the checkpoint path and a GPU. (3) MUSE fine-tuned run via `patch_musebench_lora.py` —
verified end-to-end today (dry-run + real patch + `py_compile` on all 10 patched files, all
pass) but the actual GPU run + re-parse loop hasn't executed:
```
export QWEN2_5_OMNI_LOCAL_DIR=Qwen/Qwen2.5-Omni-7B   # or wherever it's cached on the H100 box
python gpu/patch_musebench_lora.py --muse-dir <muse_dir> \
    --lora-checkpoint <Track C checkpoint path> --tag trackc \
    --out-dir <muse_dir>/Qwen2.5-Omni-LORA-trackc --dry-run   # confirm every file says [OK] first
python gpu/patch_musebench_lora.py --muse-dir <muse_dir> \
    --lora-checkpoint <Track C checkpoint path> --tag trackc \
    --out-dir <muse_dir>/Qwen2.5-Omni-LORA-trackc              # write for real
mkdir -p <muse_dir>/logs_trackc && cd <muse_dir>/logs_trackc   # MUSE's own scripts log to cwd
for f in ../Qwen2.5-Omni-LORA-trackc/*.py; do python "$f"; done
cd <path to this repo>/experiments
python gpu/parse_musebench_qwen.py --muse-dir <muse_dir> --log-dir <muse_dir>/logs_trackc \
    --model-suffix Qwen2.5-Omni-LORA-trackc \
    --out results/external_benchmarks/muse_qwen25omni-trackc.csv
# then diff acc columns against muse_qwen25omni.csv (baseline) per task
```
(4) Track E/D-zoom's checkpoints through the SAME plain harnesses as (2), clearly labeled as
"does the checkpoint hold up off-distribution" rather than "does the front-end generalize" —
lowest priority, easy to misinterpret if not labeled carefully.

## 4. After all three land

Combine with:
- this project's own battery (Track A/B/C-Z results, PAPER.md)
- MUSE Benchmark (`experiments/results/external_benchmarks/muse_qwen25omni.csv`,
  already done)
- published leaderboard citations (`BENCHMARK_LANDSCAPE.md` §2)

into one domain-organized table (pitch / harmony / rhythm / technique / genre) —
the concrete artifact that answers the mentor's question in one place. Not built
yet; do this once CMI-Bench and PitchBench numbers exist, since the whole point is
comparing our own battery's per-domain read against the same domains measured by
someone else's stimuli and questions.
