# Music Understanding Experiments

Working implementation of the study in `../RESEARCH_PLAN.md`, aimed at three
questions: **what can these models do, what can't they, and why** — with the WHY
answered by the L1 (signal) / L2 (representation) / L3 (behavior) dissociation.

- `TASKS.md` — the tier list of questions, data sources, controls, sample sizes.
- `genmodel/GENERATION_PROBES.md` — the mirror battery for *generation* models.

## Layout

```
musicprobe/            the Python package
  config.py            paths, soundfonts, instruments, sample rate
  theory.py            notes/intervals/modes/chords/keys — ground-truth vocab
  synth.py             fluidsynth rendering + numpy tone/click synthesis
  generators/          one module per stimulus family (Tier 1–2, all synthetic)
  manifest.py          stimuli.parquet schema (one row per stimulus)
  prompts.py           paraphrases + diagnostic MCQ distractors + permutation
  jobs.py              manifest -> jobs.parquet (no-audio/wrong-audio/explain jobs)
  runners/run_api.py   Track A, API models: dry | OpenAI | Gemini (temp 0, resumable)
  runners/run_local.py Track A, open models on the H100 box (Qwen2-Audio done;
                       AF3-hf / Music Flamingo-hf / Qwen-Omni loaders written,
                       hardware-unverified — runbook smoke-tests each first)
  scoring.py           parse -> accuracy/audio_gain/confusions/psychometric curves
  l1_baselines.py      classical DSP floor (pitch, cents, tempo, key)
scripts/               numbered entry points (incl. listening page + review export)
gpu/                   Track B on the H100 box: activation extraction (MERT /
                       Whisper-enc / CLAP = the encoder-family comparison),
                       probes, and attention_audio.py (does the LM attend to
                       the audio tokens, and does it decay over generation?)
genmodel/              generation-model battery: run_musicgen.py + score_gen.py
assets/soundfonts/     3 GM soundfonts (timbre held-out splits)
stimuli/  manifests/   generated artifacts
results/               tracked outputs, organized by track:
  trackA/              behavioral runs: responses__/scored__ parquets,
                       review__<model>/ CSVs, analysis_workbook.xlsx
  trackB/              probes/ (layer-wise linear probe curves per encoder)
                       + attention/ (audio-token attention diagnostics)
  genmodel/            adherence__/generations__ measurements
  l1_baseline.parquet  the shared DSP floor
```

## Run order

```bash
cd experiments

# 1. stimuli (~1200 clips, ~12 min; --quick for a 1-min smoke test)
.venv/bin/python scripts/01_generate_stimuli.py

# 2. eval jobs: manifests/jobs.parquet is COMMITTED — never rebuild it once
#    responses exist (jobs.build_jobs refuses; it would orphan all results).
#    Grow the battery with jobs.append_jobs() — see scripts/07 for the pattern.

# 2b. audit stimuli by ear (blind: answers hidden behind a click)
.venv/bin/python scripts/03_listening_page.py && open listening.html

# 3. sanity-check plumbing end to end (constant-answer backend)
.venv/bin/python -m musicprobe.runners.run_api --model dry
.venv/bin/python -m musicprobe.scoring --model dry

# 4. L1 floor: is ground truth recoverable from our own audio?
.venv/bin/python -m musicprobe.l1_baselines

# 5. real models (needs GEMINI_API_KEY)
.venv/bin/python -m musicprobe.runners.run_api --model gemini-2.5-flash --limit 100
.venv/bin/python -m musicprobe.scoring --model gemini-2.5-flash

# 6. export EVERYTHING (full verbatim responses) for manual verification
.venv/bin/python scripts/04_export_for_review.py --model gemini-2.5-flash
```

### Running everything that's left (handover for Sethu — updated 2026-08-12)

Everything through Tracks A–H, the L–Q/R–W 12-representation ladder, and the
mel-spectrogram/desk-research next actions is done and committed (full
detail: PROJECT_STATE.md "Next actions" 1–20). **What's actually queued now**
is below, in the order it's worth doing it — items 1–3 have no external
blocker and can start immediately; items 4–6 need a quick disk-availability
check first; item 7 is blocked on a Zenodo access request Rupali submitted
2026-08-12 (unknown turnaround — check back, don't wait idle); item 8 needs
manual per-repo integration work before it's runnable at all.

Full scientific rationale for every item lives in PROJECT_STATE.md's
next-actions 17–25 — this section is deliberately just "what to type," not
"why." If you hit something ambiguous, that file (and the docstring at the
top of whichever script you're running) is where the reasoning lives.

**1. Real-timbre D-zoom eval (next action 24) — ready now, closes an open
question in `PAPER.md`'s new "is D-zoom test-time scalable" section:**

```bash
pip install datasets   # only new dependency beyond the usual GPU stack

# audio + images are gitignored (stimuli/ policy, same as the whole synthetic
# battery) -- regenerate on THIS box, don't expect the committed .parquet
# files to line up with audio you don't have yet:
python -m musicprobe.real_music_nsynth --n 60 --seed 0        # ~2-5 min, streams from HF
python -m musicprobe.real_music_nsynth --dzoom-jobs --seed 0  # renders 120 PNGs, ~1 min, CPU

python gpu/eval_track_dzoom_real.py --seed 0 --limit 40  # smoke test first
python gpu/eval_track_dzoom_real.py --seed 0              # full run, 1440 jobs
```

GOTCHA: NSynth's streaming fetch is NOT perfectly reproducible run-to-run —
confirmed by running it twice on the laptop with the identical seed and
getting two different draws of notes (HF's streaming order isn't fully
stable). The `manifests/real_nsynth*.parquet` files already in git reference
audio filenames from *my* laptop run, which won't exist here — the two
commands above will overwrite them with *your* own equally-valid 180 notes.
That's expected, not a bug; just don't try to make local audio match the
committed manifest, regenerate both together.

Checkpoint: `gpu/track_d_force_ckpt/qwen25omni-zoom-s0/` (Track D-zoom is
`train_track_d_force.py --image-kind f0zoom`, not a separate file) — should
already exist from the original 2026-07-29 run. If it's missing, the eval
script fails with a clear message telling you to retrain first
(`python gpu/train_track_d_force.py --seed 0 --image-kind f0zoom`) — do NOT
evaluate an untrained checkpoint, it'll look like a real null and isn't one.

Score the output (`musicprobe.scoring` or `gpu/analyze_track_d.py`, same as
any other track) and compare against `results/trackA/trackd_zoom_summary.csv`
(the original synthetic numbers: cents 0.55→0.94, tuning 0.53→0.89). This
number is the actual answer to "does D-zoom generalize past clean synthetic
tones" — the whole reason this next action exists.

**2. Tracks X/Y — zoom+reference combo for harmony/rhythm:**

```bash
bash scripts/19_run_tracks_xy.sh   # or step through manually, see RUNBOOK_tracks_xy.md
```
CPU groundwork (renderers, full battery render, held-out splits) already
verified — smoke-test each track before the full run, same as every prior
one.

**3. Track Z — self-transcription training objective:**

```bash
python gpu/train_track_z_transcribe.py --seed 0 --smoke-test
python gpu/train_track_z_transcribe.py --seed 0
```
The biggest lift on this list (a real multi-task loss, not just a new
prompt condition — PROJECT_STATE next action 17 has the full design). The
battery score after training is NOT the deliverable by itself — rerun
`gpu/extract_activations.py --own-encoder` against this checkpoint and
compare L2 probe accuracy (`gpu/probe.py`) to the pre-fine-tune baseline;
`evaluate()` prints this reminder itself at the end of its own run.

**4. Next action 19 — nonlinear decoder per encoder layer:**

```bash
python gpu/probe_mlp.py --acts acts/<existing_dir> --task <task> --target ground_truth
```
Only tested against synthetic fake activations on the laptop — this is its
first real run, watch it. **Check first** that the raw `.npz` files from the
earlier Track B extraction (`acts/mert330`, `acts/whisper`, `acts/clap`,
`acts/qwen25omni_own`, etc.) are still on this box — if cleared, rerun
`extract_activations.py` first (commands in the quickstart section below).

**5. Next action 22 — attention audit (why do L–Q/R–W images HURT
`key_id`/`tempo_bpm`?):**

```bash
python gpu/attention_audio.py --model Qwen/Qwen2.5-Omni-7B \
  --lora-checkpoint gpu/track_l_chroma_picked_ckpt/qwen25omni-chroma-picked-s0 \
  --tag track-l-s0
```
**Check first**: are the L–Q/R–W checkpoints (`gpu/track_<l/m/.../w>_..._ckpt/`)
still on disk? If cleared, this needs at least one seed retrained before the
diagnostic can attach (`gpu/train_track_repr.py --track L --seed 0`) —
retraining just for this diagnostic is real cost, check first.

**6. Next action 25 — late-layer-loss vs. never-captured vs. nonlinear-only
diagnostic** (this is the direct answer to "is it a late-layer issue or is
it never captured" — built 2026-08-12):

```bash
# extract_activations.py needed no changes -- already takes --manifest:
python gpu/extract_activations.py --model Qwen/Qwen2.5-Omni-7B --own-encoder \
  --manifest manifests/real_nsynth_manifest.parquet --out acts/qwen25omni_own_real_nsynth

python gpu/probe.py --acts acts/qwen25omni_own_real_nsynth --task pitch_note_id \
  --target ground_truth --group-key instrument_family
python gpu/probe_mlp.py --acts acts/qwen25omni_own_real_nsynth --task pitch_note_id \
  --target ground_truth --group-key instrument_family

python gpu/classify_layer_pattern.py \
  --linear results/trackB/probes/probe__acts_qwen25omni_own_real_nsynth__pitch_note_id__ground_truth.csv \
  --nonlinear results/trackB/probes/probe_mlp__acts_qwen25omni_own_real_nsynth__pitch_note_id__ground_truth.csv
```
Run per task (`key_id`, `mode_id`, `interval_id`, `chord_quality`, etc.) —
the verdict may genuinely differ task to task, that's expected and is the
actual point (Rupali's framing: "maybe it varies test to test") — report
every task's verdict, don't average into one number. Five possible verdicts,
explained in the script's own docstring: `NEVER_CAPTURED`, `LATE_LAYER_LOSS`,
`PRESENT_THROUGHOUT`, `NONLINEAR_ONLY`, `MIXED`.

**7. Next action 23 — MedleyDB (blocked on Zenodo access):**

Rupali requested access 2026-08-12 (zenodo.org/record/2628782) — unknown
turnaround, check back. **When it lands**: download `MedleyDB-Melody.zip`,
unzip anywhere, then:
```bash
python -m musicprobe.real_music_medleydb --data-home /path/to/MedleyDB-Melody
```
builds the manifest + jobs (segmentation, ground truth, `wrong_audio`
contamination control, `factors.track_id` for probing — all already built
and verified against a synthetic stand-in). **No eval-only script exists for
this yet** (unlike NSynth above) — build one following
`gpu/eval_track_dzoom_real.py`'s exact pattern (same checkpoint, same image
rendering call, just point `held`/jobs at this module's output instead).
Also worth doing once real audio is in hand: mirdata's `melody3` annotation
(ALL simultaneous melodic lines, not just the predominant one) is the right
ground truth for actually testing PAPER.md's new "D-zoom is monophonic-only"
claim properly, rather than the one anecdotal real-recording test it's
currently based on — `melody1` (what's used now) matches D-zoom's own
monophonic `pyin` front-end, but can't test polyphonic content at all. Not
built; flag to Rupali if you get this far and want to scope it.

**8. Next action 21 — MU-LLaMA / MusiLingo / M2UGen (lowest priority):**

`gpu/eval_music_lalms.py` has the harness (job iteration, resumable, same
output schema as everything else) but the three `_load_*` functions are
`NotImplementedError` stubs on purpose, not an oversight — port each from
that repo's own demo script (URLs in the file's docstring) rather than
guessing at their API. These three all sit on frozen MERT, so this answers
roughly the same question as next action 19 at much higher setup cost — do
19 first; if it already closes the gap on MERT's near-floor tasks, this adds
less than it looks like it would.

**Consolidated gotchas (things that already bit someone once, so you don't
have to rediscover them):**
- **Eager attention is required.** `attention_audio.py` hard-fails via
  `assert_eager_attention()` if a model silently falls back to `sdpa` — a
  2026-07-24 run without this check produced numbers that had to be
  retracted. Don't bypass it.
- **`extract_activations.py --own-encoder`'s submodule path is a best
  guess** — it prints the path it found; double-check against the real
  architecture the first time you use it for a given model family.
- **LoRA never touches `audio_tower`, on purpose.** Every Track C/D/E/F/Z
  `target_modules` regex only matches under `thinker.<lm_path>` — confirmed,
  not a bug, and it's WHY D-zoom/E are "substitution, not hearing" rather
  than improved perception.
- **`_held_out_mask` has three tiers for a reason** (soundfont → base_midi →
  bpm-quantile). Tasks with neither of the first two silently got a 0-row
  held-out split before the third tier existed — if you ever add a new task
  cluster, verify the split produces nonzero train/held before spending GPU
  time on it.
- **Real-music manifests have no `soundfont`.** Pass `--group-key track_id`
  (MedleyDB) or `--group-key instrument_family` (NSynth) to
  `probe.py`/`probe_mlp.py` — without it, every real-music row collapses
  into one group and you silently get a meaningless random split instead of
  a real held-out fold.
- **`jobs.parquet` is frozen** — `musicprobe.jobs.build_jobs` refuses to
  rebuild it on purpose (would orphan every existing result). Use
  `append_jobs()` if a new task is ever added to the v1 battery.

**Reporting back:**
```bash
git add results/ manifests/*.parquet
git commit -m "H100 run: <what you ran>"
git push
```
Then update the relevant next-action entry in PROJECT_STATE.md with what
happened (status, numbers, any new gotcha) — that file is the single source
of truth for "what's been tried," per its own header instruction to update
it every time something changes hands.

`11_run_track_c.sh` / `13_run_track_d.sh` below are now historical — kept
for reference, not something you need to run again.

<details>
<summary>Legacy: the original full-battery runbook (mostly superseded, kept for reference)</summary>

```bash
bash scripts/09_smoke_test.sh     # preflight (~2 min, no GPU/API cost): fix
                                  # anything it flags, rerun until it passes
bash scripts/08_run_remaining.sh  # then the real thing
```

One script, dependency-ordered, resumable (rerun it after any crash — finished
work is skipped), logs to results/runlogs/. It covers: instrument_id top-ups
for Qwen2-Audio + Gemini, the remaining Track-A models, scoring + review +
workbook exports, all three Track-B encoder extractions + the full probe
suite + the attention diagnostic, the MusicGen battery, and the final
results/ commit+push. One-time prereqs are in the script header; PORTKEY_API_KEY
is REQUIRED (the runbook stops rather than silently skipping API work). All
five local models run on plain transformers with hub-verified checkpoint ids
— nothing to install by hand beyond the pip line. (GPT-4o-audio dropped
2026-07-25 — no OpenAI API access; OPENAI_API_KEY is no longer checked or
used anywhere in this runbook.) **Rerunning this now would redo already-
committed work** (it's resumable so it won't corrupt anything, but it will
burn GPU/API time re-verifying finished steps, including re-running the
attention diagnostic loop inside this script, which is otherwise done) — only
use it if you specifically need to redo the full battery from scratch.

</details>

### On the H100 box — quickstart

```bash
git clone <this repo> && cd Music_Understanding/experiments

# one-time setup (audio WAVs are not in git — regenerate them, ~12 min):
sudo apt-get install -y fluidsynth          # or: conda install -c conda-forge fluidsynth
pip install -r requirements.txt
bash scripts/00_download_soundfonts.sh
python scripts/01_generate_stimuli.py       # deterministic: seeded, stable hashing —
                                            # reproduces exactly the committed manifest
# jobs.parquet is committed — use as-is (rebuilding is disabled while results exist)

# sanity check before burning GPU time:
python -m musicprobe.runners.run_api --model dry && python -m musicprobe.scoring --model dry
```

Then (results/ is tracked in git — commit outputs there and push):

```bash
pip install torch transformers accelerate soundfile pandas pyarrow librosa
# Track A, open models — same jobs file, same output format:
python -m musicprobe.runners.run_local --model Qwen/Qwen2-Audio-7B-Instruct
# Track B, encoder-family comparison (music-SSL vs ASR vs contrastive):
python gpu/extract_activations.py --model m-a-p/MERT-v1-330M --out acts/mert330
python gpu/extract_activations.py --model openai/whisper-large-v3 --out acts/whisper
python gpu/extract_activations.py --model laion/clap-htsat-unfused --out acts/clap
python gpu/probe.py --acts acts/mert330 --task key_id --target ground_truth
# Generation-model battery:
python genmodel/run_musicgen.py --model facebook/musicgen-medium
python genmodel/score_gen.py --dir genmodel/outputs

# package everything for review and push back:
python scripts/04_export_for_review.py --model Qwen/Qwen2-Audio-7B-Instruct
git add results/ genmodel/outputs/adherence.csv genmodel/outputs/generations.parquet acts/*/probe*.csv 2>/dev/null
git commit -m "model run outputs" && git push
```
(Generated WAVs in genmodel/outputs/ are gitignored — share those via Drive/scp
if listening is needed; adherence.csv carries the measurements.)

Then send `results/` (and `genmodel/outputs/adherence.csv`) back for analysis —
every response is stored verbatim; `scripts/04_export_for_review.py` turns a
run into per-task CSVs (full prompts + full raw responses + audio paths) plus
`explanations.csv` for the manual listening-vs-guessing analysis.

### Model roster (who gets tested, where)

| Model | How | What |
|---|---|---|
| Gemini 2.x | `run_api.py` (laptop) | full battery — behavioral ceiling (GPT-4o-audio out of scope, no OpenAI API access) |
| Qwen2-Audio-7B | `run_local.py` (H100) | full battery — harness-validation anchor (replicate its published MMAU-music score first) |
| AF3-hf, Music Flamingo-2601-hf, Qwen2.5/3-Omni | `run_local.py` (loaders written, unverified) | full battery + Track B layer-wise probes |
| MERT / Whisper-enc / CLAP | `gpu/extract_activations.py` | Track B only — encoder-family comparison |
| MusicGen (Stable Audio Open later; Suno/Udio manual) | `genmodel/` (H100) | constraint-adherence battery |
| Claude | API, text only | symbolic ABC-notation contrast arm (specced, task 3.7) |

## Interpreting output (the short version)

`scoring.summary_table` prints per task: `acc_audio`, `acc_no_audio`,
`audio_gain`, unparseable rates. **`audio_gain` ≈ 0 means the task measured text
priors, not hearing** — that cell is invalid regardless of accuracy. Confusion
matrices (`scoring.confusion`) separate musically-structured errors from
guessing; `scoring.psychometric_cents` gives the pitch-resolution threshold;
`scoring.tempo_errors` gives the octave-error histogram. The L2/L3 dissociation
table = colab/probe.py accuracies joined against behavioral accuracies on the
same stimuli.

## Current status (2026-07-17)

- ✅ Tier 1–2 synthetic battery implemented and smoke-tested end to end
  (generate → jobs → dry run → score); L1 floor: pitch 100%, cents 100%,
  tempo 75% (n=4, quick), key 62% (naive Krumhansl struggles on minor cadences —
  use essentia/madmom on Colab for the real L1 on cadence/real-music stimuli).
- ✅ fluidsynth 2.5.6 + 3 soundfonts installed; venv at `experiments/.venv`.
- ✅ 2026-07-17 discussion round: meter task reframed as beats-per-bar (audio
  can't convey the notated denominator), key "cadence" form renamed
  "progression", added tuning_judgment (12-TET probe), explain-format jobs,
  listening.html audit page, full-response review exports, H100 local runner,
  MusicGen adherence battery scripts.
- ✅ 2026-07-19: Qwen2-Audio-7B + Gemini-2.5-pro full runs landed (PR #1);
  refusal-aware scoring (refusing without audio counts as incorrect, reported
  as refused_*); analysis workbook export (scripts/06).
- ✅ 2026-07-20: instrument_id task added over the pitch clips (jobs appended,
  existing job rows untouched — reruns stay resumable); results/ reorganized
  into trackA/trackB/genmodel; loaders written (UNVERIFIED) for Qwen2.5-Omni,
  Qwen3-Omni, AF3, Music Flamingo; gpu/attention_audio.py (audio-token
  attention mass + decay); scripts/08_run_remaining.sh = the one-shot H100
  runbook for everything below.
- ✅ 2026-07-22 to 07-24: remaining Track-A models + MOSS-Music-8B landed;
  Track B baseline probes (MERT/Whisper/CLAP) + own-encoder re-probe landed;
  MusicGen battery landed; Track C (3-arm LoRA on AF3) and Track D Phase 1
  (Qwen2.5-Omni-7B + spectrogram-image, RESEARCH_PLAN.md §12) built and
  ready to run.
- ✅ 2026-07-25: attention diagnostic re-verified correctly on all 5 open
  models (commit c348ea6, eager-attention-verified — the 2026-07-24 run was
  retracted, see PROJECT_STATE.md Known gaps); microtone probe (relative
  pitch direction vs. absolute tuning, `gpu/probe_microtone.py`) landed;
  GPT-4o-audio dropped from the plan entirely (no API access).
- 📋 Next: Track C and Track D Phase 1 — see "Running everything that's left"
  above. Qwen2-Audio MMAU-music replication (harness validation) still
  pending, lower priority.
- 📋 Tier 3 (VocalSet/GuitarSet/Jamendo): specced, not implemented.
