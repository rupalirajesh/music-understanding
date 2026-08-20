# What to do next — Claude Code handoff

## The project in one sentence

Build and evaluate an **evidence-grounded music analyst**: a system that can
answer music questions while clearly separating (1) sourced music knowledge,
(2) what it measured in the submitted recording, and (3) the theory inference
made from those measurements.

This is **not** a claim that a model should replace musicologists, cover every
musical culture, or infer every rich musical fact from an arbitrary recording.
The contribution is to test whether an answer is genuinely grounded in audio
and tools, rather than merely plausible text-prior recall.

Read these first:

- `DIRECTION_DECISION.md` — paper direction and the eventual three-suite evaluation.
- `TONIGHT_AZURE_PLAN.md` — prior runbook and current experiment context.
- `PROJECT_STATE.md` — existing results/status. Treat past null results as diagnostic context, not a final conclusion.
- `experiments/TASKS.md` and `experiments/README.md` — experiment commands and layout.

## Research claim to protect

> Current music audio-language models can sound knowledgeable without being
> audio-grounded. We measure that gap, localize the failure, and test whether
> auditable music tools—not just another undifferentiated visual modality—make
> musical analysis reliable and transferable.

Every model answer should ideally distinguish:

| Answer component | Allowed evidence |
| --- | --- |
| Known fact about a named song | Retrieved/cited source |
| Property of this uploaded performance | Audio measurement, timestamp, uncertainty |
| Musical interpretation | Explicit theory rule applied to the evidence |

Do not treat a model's statement about its own reasoning as provenance. Record
tool outputs, timestamps, source IDs, and controlled swaps instead.

## Cost-safe first session (RunPod + VS Code + Claude Code)

Use a **RunPod Pod**, not Serverless. Choose one `A100 80 GB` first. It is
enough for the existing Qwen/LoRA experiments and is materially cheaper than
an H100. At the currently listed community-cloud price of about **$1.39/hour**,
the setup plus smoke tests should be roughly **$3–$8** if stopped within 2–6
hours. This is an estimate, not a quote: availability and host price vary.

Start with **$15 of RunPod credit**, turn off automatic spending if offered,
and stop the Pod whenever not actively running. A full multi-seed training
study is a later decision, not tonight's commitment. Budget roughly $10–$25
per successful one-seed training/evaluation attempt only after timing a smoke
test; do not promise runtime before measuring it.

Claude Code usage/billing is separate from RunPod GPU charges. It uses the
user's Anthropic account/plan while it operates in VS Code; a long agentic
session can consume that plan's allowance even if the Pod is idle.

Recommended Pod settings:

- GPU: 1 × A100 80 GB. Use an H100 only if its price is close and it is readily available.
- Template: official PyTorch template.
- Disk: 30–50 GB container disk plus a **100 GB persistent volume** mounted at `/workspace`.
- Enable: **SSH Terminal Access**.
- Region: choose an available one; capacity matters more than location for this first run.

RunPod's `/workspace` volume survives a stopped Pod; the ordinary container
disk does not. Stop (do not terminate) when pausing. Terminate only after
copying anything important from the volume.

## Connect VS Code to the Pod

1. Before deploying the Pod, add the local machine's public SSH key to the RunPod account.
2. Create the Pod with SSH Terminal Access enabled.
3. In RunPod, copy the SSH command from **Connect**. In VS Code, install the
   **Remote - SSH** extension and add that host to its SSH configuration.
4. Open a Remote-SSH VS Code window, connect as `root`, and open `/workspace`.
5. Install and sign into Claude Code **on the remote Pod**, then run it from
   the repository folder:

```bash
npm install -g @anthropic-ai/claude-code
claude
```

If Node is absent, install a current Node LTS first. Do not use `sudo` for the
global Claude Code install. Authenticate with the user's own Anthropic account.

Useful official references: RunPod's [pricing](https://www.runpod.io/pricing),
[VS Code connection guide](https://docs.runpod.io/pods/configuration/connect-to-ide),
and [Pod lifecycle/billing guide](https://docs.runpod.io/pods/manage-pods);
Anthropic's [Claude Code setup guide](https://docs.anthropic.com/en/docs/claude-code/getting-started).

## Repository/environment setup

First inspect the remote repository and do not overwrite user work:

```bash
cd /workspace
git clone https://github.com/rupalirajesh/music-understanding.git music-understanding
cd /workspace/music-understanding
git checkout rupali-music
git status
```

Work exclusively on the `rupali-music` branch, not `main`. This project has a second
contributor (Sethu) who pushes directly to `main`; `rupali-music` is a dedicated
branch so this work never collides with or depends on his pushes. Push only to
`rupali-music` (`git push origin rupali-music`) — never push to `main`.

Do **not** stage, commit, discard, or reformat unrelated changes.

Install the audio/system dependency and Python environment. Adapt only if the
repository's existing requirements say otherwise.

```bash
apt-get update
apt-get install -y fluidsynth

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers accelerate peft datasets soundfile librosa
python -m pip install -r requirements.txt

nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The source audio files are deliberately ignored by Git. Regenerate them from
the tracked generator, then verify the frozen manifest was not altered:

```bash
cd /workspace/music-understanding/experiments
bash scripts/00_download_soundfonts.sh
python scripts/01_generate_stimuli.py
python scripts/05_selftest.py
git diff --exit-code -- manifests/stimuli.parquet
```

Do **not** regenerate or edit `jobs.parquet`; it is frozen. If the manifest
differs, stop and report the diff rather than silently accepting it.

## Execute in this order

### 0. Record the machine and establish a smoke-test log

Create a timestamped run note in an ignored `results/` directory containing:

- GPU model, driver/CUDA, Python/Torch versions, model revision;
- the exact command and start/end time for every run;
- raw model output for at least five items per condition;
- any failed command and its full error.

Do not summarize a result as a paper finding from five items. A smoke test only
checks that the pipeline is real, GPU-backed, and that outputs are parseable.

### 1. Run external PitchBench first

This is the quickest check of the current model/evaluation environment. Read
the script's `--help` first and use its current required arguments. Run a tiny
smoke test into a fresh output directory, inspect five raw examples, then run
a modest capped evaluation only if the smoke result is valid.

Expected entry point:

```bash
python gpu/eval_pitchbench.py --help
python gpu/eval_pitchbench.py \
  --config pitchbench_a1_single_pitch_id \
  --limit 5 \
  --out-dir results/external_benchmarks/pitchbench_smoke
```

After checking examples, use a **new** output directory for a larger capped
run (for example, `--limit 100`), because resumable scripts may skip items in
an existing directory. Save both outputs. Do not report an accuracy without
checking answer parsing and individual examples.

### 2. Run the Track Z transcription-auxiliary smoke test

The strongest existing intervention to test is **Track Z**: the answer task
plus an auxiliary self-transcription target (onset, duration, frequency JSON).
It tests an operational hypothesis: forcing an auditable intermediate music
representation may improve transfer beyond generic spectrogram images.

Run only the built-in smoke test tonight:

```bash
python gpu/train_track_z_transcribe.py --help
python gpu/train_track_z_transcribe.py --seed 0 --smoke-test
```

Success criteria: model loads on GPU, one training step completes, loss is
finite, a checkpoint/output is produced, and the answer-only evaluation path
can load the result. Record actual wall time and GPU memory. **Do not launch
a full training run or multiple seeds without showing the user the smoke
result and expected cost first.**

### 3. Prepare CMI-Bench; smoke it only if setup is clean

CMI-Bench is an external MIR instruction-following benchmark, useful as a
broad transfer diagnostic. Its full run is not a substitute for the proposed
grounding evaluation and may take time/data downloads.

1. Read `experiments/scripts/RUNBOOK_external_benchmarks.md`.
2. Follow its pinned download/clone instructions exactly.
3. Run the evaluation script's `--help` and a five-item smoke test to a new
   result folder.
4. Inspect raw answers before any larger run.

Expected project entry point:

```bash
python gpu/eval_cmibench.py --help
```

If a required external asset, credential, model weight, or licence is missing,
stop and report the precise blocker. Do not replace the benchmark with a
differently sourced dataset without permission.

### Do not spend time on these tonight

- Do not run the generic visual-DSP ladder (Tracks X/Y). Existing work suggests
  generic spectrogram/chromagram/rhythm images are neutral or harmful; that is
  a useful boundary condition, not the desired headline.
- Do not claim that F0 zoom or contour reading proves general “music
  understanding.” It is a narrow perception capability.
- Do not build an all-genres/all-cultures/all-difficulties benchmark. That is
  not credible or feasible in this phase.
- Do not download commercial music as a training/evaluation corpus or assume a
  short clip is automatically usable. Keep provenance and licensing explicit.

## Next implementation task (plan first; do not silently build it)

After the above smoke tests, propose a small **grounding pilot**, not a huge
benchmark. The initial deliverable is an implementation plan and sample items
for user approval.

Create perhaps 24 paired, controlled examples (not commercial music) across
four phenomena:

1. pitch/motif transposition;
2. syncopation versus straight rhythm;
3. tempo or metric change;
4. simple harmonic-function or modulation change.

For each item, create only one targeted change at a time and test:

| Condition | What it tests |
| --- | --- |
| audio absent | text-prior baseline |
| correct audio | normal answer |
| swapped audio | whether answer changes with the sound |
| correct audio + tool report | benefit of an auditable measurement |
| correct audio + swapped tool report | whether tool evidence is actually used |

Required structured output:

```text
claim:
evidence_span_seconds:
measurement_or_tool_output:
theory_rule_or_source_id:
confidence:
```

The point is not that an F0 contour alone constitutes understanding. The point
is a falsifiable grounding test: the answer should follow the correct audio and
the correct evidence, and fail appropriately when either is swapped.

For named-song questions, a correct answer may legitimately combine retrieved
knowledge (with a source) and recording-specific measurement. The experiment
must label which portion came from which source.

## First prompt to give Claude Code

Paste this after opening the remote repository:

```text
Read WHAT_TO_DO.md, DIRECTION_DECISION.md, PROJECT_STATE.md, and the relevant
experiment README/runbook before changing anything. Work only on the setup and
the numbered smoke tests in WHAT_TO_DO.md. First show me: (1) git status, (2)
which exact requirements/dependencies are needed, and (3) the exact smoke-test
commands you intend to run. Preserve existing work, never alter frozen
manifests/jobs, save raw outputs and run metadata, and stop after the Track Z
smoke test to report timing, GPU memory, errors, and five raw outputs. Do not
launch full training, download substitute datasets, or run Tracks X/Y without
my explicit approval.
```

## What a useful check-in looks like

At each stopping point report only:

1. completed command(s) and elapsed time;
2. GPU/RAM peak and approximate projected cost from measured rate;
3. five raw input/output examples and parser status;
4. exact files saved;
5. the one decision needed before more spending.

That makes the next step scientific and auditable, rather than a long,
uninterpretable GPU run.
