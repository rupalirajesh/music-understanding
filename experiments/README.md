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
                       AF3 / Music Flamingo / Qwen-Omni loaders TODO)
  scoring.py           parse -> accuracy/audio_gain/confusions/psychometric curves
  l1_baselines.py      classical DSP floor (pitch, cents, tempo, key)
scripts/               numbered entry points (incl. listening page + review export)
gpu/                   Track B on the H100 box: activation extraction (MERT /
                       Whisper-enc / CLAP = the encoder-family comparison) + probes
genmodel/              generation-model battery: run_musicgen.py + score_gen.py
assets/soundfonts/     3 GM soundfonts (timbre held-out splits)
stimuli/  manifests/  results/    generated artifacts
```

## Run order

```bash
cd experiments

# 1. stimuli (~1200 clips, ~12 min; --quick for a 1-min smoke test)
.venv/bin/python scripts/01_generate_stimuli.py

# 2. expand into eval jobs with controls
.venv/bin/python scripts/02_build_jobs.py

# 2b. audit stimuli by ear (blind: answers hidden behind a click)
.venv/bin/python scripts/03_listening_page.py && open listening.html

# 3. sanity-check plumbing end to end (constant-answer backend)
.venv/bin/python -m musicprobe.runners.run_api --model dry
.venv/bin/python -m musicprobe.scoring --model dry

# 4. L1 floor: is ground truth recoverable from our own audio?
.venv/bin/python -m musicprobe.l1_baselines

# 5. real models (needs GEMINI_API_KEY / OPENAI_API_KEY)
.venv/bin/python -m musicprobe.runners.run_api --model gemini-2.5-flash --limit 100
.venv/bin/python -m musicprobe.scoring --model gemini-2.5-flash

# 6. export EVERYTHING (full verbatim responses) for manual verification
.venv/bin/python scripts/04_export_for_review.py --model gemini-2.5-flash
```

### On the H100 box — sethu quickstart

```bash
git clone <this repo> && cd Music_Understanding/experiments

# one-time setup (audio WAVs are not in git — regenerate them, ~12 min):
sudo apt-get install -y fluidsynth          # or: conda install -c conda-forge fluidsynth
pip install -r requirements.txt
bash scripts/00_download_soundfonts.sh
python scripts/01_generate_stimuli.py       # deterministic: seeded, stable hashing —
                                            # reproduces exactly the committed manifest
# do NOT rerun 02: use the committed manifests/jobs.parquet as-is

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
| Gemini 2.x, GPT-4o-audio | `run_api.py` (laptop) | full battery — behavioral ceilings |
| Qwen2-Audio-7B | `run_local.py` (H100) | full battery — harness-validation anchor (replicate its published MMAU-music score first) |
| Audio Flamingo 3, Music Flamingo, Qwen2.5/3-Omni | `run_local.py`, loaders TODO | full battery + Track B layer-wise probes |
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
- 📋 Next: Gemini/GPT-4o small pilot (~$; check per-model audio format limits);
  on H100: Qwen2-Audio replication of a published MMAU-music number (harness
  validation), MERT/Whisper/CLAP extraction + probes, MusicGen battery.
- 📋 Tier 3 (VocalSet/GuitarSet/Jamendo): specced, not implemented.
