# Handover — probing / power-test session (for the next Claude Code instance)

Written 2026-08-13 by Sethu's Claude on the old VM, for whoever (likely another
Claude Code instance) picks this up on a new VM. Read this top-to-bottom before
running anything. `PROJECT_STATE.md` is the full project history and next-actions
list; this doc is scoped to **the encoder-probing thread** and the immediate
next steps.

Latest commit when this was written: **`d655020`** (pushed to `origin/main`).

---

## 0. First thing to understand: what is and isn't on the new VM

The repo is on git, but **large regenerable artifacts are gitignored and will NOT
be on the new VM**:
- `experiments/stimuli/**` — all audio WAVs (battery + all aug sets). Regenerable, deterministic.
- `experiments/acts/`, `experiments/acts_nsynth/`, `experiments/acts_keymode/` — extracted encoder activations (`.npz`). Must be **re-extracted on the new VM** (needs a GPU).
- `experiments/gpu/*_ckpt/`, `*_checkpoints/` — trained LoRA adapters (Track C–Z, D-zoom, etc.). **Gone unless retrained.**
- HF model cache lives at `HF_HOME=/scratch/sethu/hf` — if `/scratch` is not shared to the new VM, models re-download on first use.

**Implication:** any probing/eval that reads activations or checkpoints needs a
regenerate + re-extract (or retrain) step first. Commands below.

---

## 1. Environment setup (new VM)

```bash
# repo (adjust path if cloned elsewhere)
cd /scratch/sethu/music-understanding-repo/experiments   # working dir is experiments/

# conda env (has transformers 5.14.1, torch 2.6, peft, datasets, librosa, fluidsynth)
source /home/sethu/miniconda3/etc/profile.d/conda.sh && conda activate music
# if the env doesn't exist on the new VM, recreate from experiments/requirements.txt
# (note: Qwen3-Omni needs transformers 5.x = this env; MOSS-Music needs 4.57 = separate `moss` env)

export HF_HOME=/scratch/sethu/hf            # model cache
```

**GPUs are SHARED with other people's jobs** (sglang / VLLM show up in nvidia-smi).
Only use GPUs sitting at ~0 MiB. Check first: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`.
Pin with `CUDA_VISIBLE_DEVICES=<N>`. **Do NOT kill other jobs without asking the user.**

**Push:** the repo is public (`github.com/rupalirajesh/music-understanding`), Sethu is
a collaborator. This session pushed via a token file at `/scratch/sethu/data/gh_token.env`
(read `head -n1`, strip a `export GITHUB_TOKEN=` / quotes prefix — NEVER `source` or `echo`
it). If that file isn't on the new VM, ask the user how they want to push.

---

## 2. What this session did (encoder-probing thread), newest first

All findings are about the **frozen AUDIO encoders' representations of the audio**
(MERT=music, Whisper=speech, CLAP, + each LALM's own audio tower). Not vision. Not the LLM.

| commit | what | headline |
|---|---|---|
| `d655020` | **POWER TEST** on mode/key (10–20× more data + permutation nulls) | **The "mode is absent" null was a SAMPLE-SIZE ARTIFACT.** |
| `7e90275` | `probe_seq.py` — structure-aware (attention-pool, **no mean-pooling**) decoder | attention-pool doesn't beat mean-pool on small data (overfits); mode still ~chance at small n |
| `51c5210` | layer-pattern probe on **real NSynth** instrument recordings | mechanism replicates on real audio: MERT retains absolute pitch, Whisper discards it late |
| `f3ab7ae` | nonlinear MLP probe (`probe_mlp.py`) + layer-pattern verdicts (`classify_layer_pattern.py`), synthetic | (small-data) mode=never, key/interval=late-loss, nonlinear recovers nothing linear missed |

### The key result (commit d655020) — READ THIS
The earlier "near-floor" verdicts came from the frozen battery's ~104 mode / ~96 key
stimuli — at 13/24 classes with a held-out-soundfont split, that's ~3–5 examples/class.
Sethu doubted that was enough to conclude "absent." So we generated **1824 new
mode+key stimuli** (mode 96/class, key 24/class, across all 3 soundfonts) and re-probed
with a permutation null. Result (best layer, p over 50 shuffles):

| encoder | task | small-data | **LARGE-data (peak layer)** | p |
|---|---|---|---|---|
| MERT (music) | mode_id | ~chance 0.088 | **0.278 @ L6** (+0.20) | 0.02 |
| MERT | key_id | ~0.21 | **0.913 @ L2** (+0.87) | 0.02 |
| Whisper (speech) | mode_id | ~chance | **0.248 @ L2** (+0.17) | 0.02 |
| Whisper | key_id | ~chance | **0.788 @ L0** (+0.75) | 0.02 |

**Corrected conclusions:**
1. mode AND key are decodable from both music and speech encoders given adequate n — "absent" was lack of power.
2. Signal peaks in **early layers** → "early-capture / late-discard," not "never captured." (This is *why* the LLM, which reads the final layer, struggles.)
3. **Any "task X is absent from the encoder" claim from the ~100-stimulus battery is unsafe and must be re-checked at scale.** PAPER.md's "absent/never-captured" language should be softened.

Summary CSV: `experiments/results/trackB/probes/keymode_powertest_summary.csv`.

---

## 3. IMMEDIATE next step (unfinished): Qwen2.5-Omni own-tower power re-run

This is the one loose end. During the power test, extraction of Qwen2.5-Omni's own
audio tower was **killed at 129/1824** (user needed the GPU; all GPUs later got busy).
MERT + Whisper are done; Qwen (the encoder the actual LLM reads) is not.

**Exact steps (needs one free GPU):**
```bash
cd /scratch/sethu/music-understanding-repo/experiments
source /home/sethu/miniconda3/etc/profile.d/conda.sh && conda activate music
export HF_HOME=/scratch/sethu/hf

# 1. Regenerate the mode/key WAVs (gitignored; deterministic — same as the run that made the manifest)
python scripts/generate_aug_keymode.py --mode-reps 2 --key-reps 4
#    -> writes stimuli/mode/augm_* + stimuli/key/augk_*, rebuilds manifests/aug_keymode_manifest.parquet (1824 rows)

# 2. Extract Qwen2.5-Omni own audio-tower activations (pick a FREE gpu, ~30 min for 1824 on a 7B)
CUDA_VISIBLE_DEVICES=<FREE> python gpu/extract_activations.py \
    --model Qwen/Qwen2.5-Omni-7B --own-encoder \
    --out acts_keymode/qwen25omni_own --manifest manifests/aug_keymode_manifest.parquet

# 3. Probe with the permutation-null power-test probe (CPU is fine)
python gpu/probe_keymode_powertest.py qwen25omni_own
#    prints mode_id + key_id best-layer acc, peak layer, permutation p
```
Expected (hypothesis): Qwen's own tower also recovers mode/key at scale, signal in
early layers — completing the "it's the final-layer readout, not the encoder" story
for the model we actually intervene on. Append the numbers to
`results/trackB/probes/keymode_powertest_summary.csv` and commit.

**Gotcha that bit us:** `probe.py`/`probe_mlp.py`/`probe_keymode_powertest.py` read the
DEFAULT battery manifest unless you pass the right one. `probe_keymode_powertest.py`
hardcodes `aug_keymode_manifest.parquet` (fine). For `probe.py`/`probe_mlp.py` on any
non-battery activations you MUST pass `--manifest <that manifest>` and usually
`--group-key <soundfont|instrument_family|track_id>`, and a distinct `--out` dir (the
probe's output filename is keyed on the acts dir NAME, so `acts_keymode/mert330` and
`acts/mert330` collide — use `--out results/trackB/probes_keymode` to avoid clobbering).

---

## 4. Follow-on work in the same thread (in priority order)

1. **Extend the power test to `interval_id` and `chord_quality`** (the other near-floor
   tasks) and to the other encoders (AF3, Music-Flamingo, Qwen3-Omni own towers). The
   mode/key generator (`scripts/generate_aug_keymode.py`) only makes mode/key; interval
   and chord need their own large sets — generators exist: `musicprobe/generators/intervals.py`,
   `chords.py` (write a `generate_aug_interval_chord.py` in the same spirit, across soundfonts).
2. **Add permutation-null CIs to the older probe outputs** and re-derive verdicts at scale
   — the small-data `probe__*`/`probe_mlp__*` CSVs and `classify_layer_pattern` verdicts are
   now known to be under-powered for the ≥13-class tasks.
3. **Soften PAPER.md / PROJECT_STATE.md language**: replace "absent / never-captured" for
   mode/key with "not decodable at the battery's sample size; decodable in early layers at
   scale; attenuated by the final layer." (Sethu will likely want to review before pushing prose.)

## 5. Larger queued items (from coauthor's PROJECT_STATE next-actions — separate thread)

These are built/scaffolded but GPU-unverified; they need the trained checkpoints
(gitignored → likely need retraining on the new VM):
- **Track Z (self-transcription)** — `gpu/train_track_z_transcribe.py`. The one intervention
  where the model builds its OWN representation; the most promising "does it learn to hear"
  test. Directly motivated by the early-capture/late-discard finding above (force the encoder
  to preserve, to the output, what it currently discards).
- **#24 real-timbre D-zoom/E eval** — manifests `real_nsynth_jobs.parquet` +
  `real_nsynth_dzoom_jobs.parquet` and script `gpu/eval_track_dzoom_real.py` are READY; needs
  the trained D-zoom (`track_d_force_ckpt` / f0zoom) and E (`track_e_ckpt`) checkpoints
  (gitignored → retrain via `gpu/train_track_d_force.py --image-kind f0zoom` and
  `gpu/train_track_e_f0text.py`).
- **Tracks X/Y (zoom+reference COMBINATION for harmony/rhythm)** — `scripts/19_run_tracks_xy.sh`,
  registered in `gpu/train_track_repr.py`. The untried combination (L–Q/R–W tested zoom and
  reference separately, both null).
- **MuChoMusic** — deferred (metadata-only HF set; audio is external SDD+MusicCaps/YouTube;
  see `gpu/eval_muchomusic.py` BLOCKER section).

---

## 6. Overall project state in one paragraph (context for the whole arc)

Audio-LLMs fail at fine microtonal pitch and several harmony/rhythm tasks. Diagnosis
across L1(DSP)/L2(encoder probe)/L3(behavior): **relative pitch is present in encoders
but under-used; absolute tuning is genuinely hard.** Interventions: fine-tuning can't buy
microtones back (Track C); a **spectrogram image is ignored** (Track D conclusive); a
**zoomed pitch chart with a reference line fixes both relative AND absolute pitch**
(D-zoom, cents 0.55→0.94, tuning 0.53→0.89) but it's **substitution** (a DSP tracker does
the perceiving, the model reads the chart, ignores audio); **F0-as-text** fixes relative
pitch scalably; **learned pitch-fusion is a null**; **harmony/rhythm visual representations
(Tracks G, L–W) don't transfer** (pitch is special). Newest thread (this session): the
**encoder-probing "absent" verdicts were under-powered** — mode/key ARE encoded (early
layers) at scale; the readout/late-layer is the bottleneck. Oral-paper direction discussed:
pivot to real music + a general perception adapter + prove complement-not-substitute.
