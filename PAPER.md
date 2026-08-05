# Paper draft skeleton — living doc

Working title: *Where Music Understanding Breaks: Localizing Perception,
Alignment, and Reasoning Failures in Audio Language Models*

## Research questions
1. **Capability**: which musical tasks can current audio LMs do, at what
   precision? (tiered battery: perception → structure → semantics)
2. **Mechanism**: for each failure, is it perception (encoder), alignment
   (encoder→LM bridge), or reasoning (LM) — and is it fixable by fine-tuning?
3. **Representation**: does providing explicit representations (in-context
   DSP features, symbolic notation) close the gaps — and what does that imply
   for a universal, genre-portable music representation?
4. Same questions for music *generation* models, via constraint adherence.

## Method (one line each)
- 1,176 synthetic stimuli, perfect ground truth (MIDI+fluidsynth, 3
  soundfonts, 4 instruments; numpy synthesis for microtonal).
- 2,063 questions with controls: no-audio twins (text-prior detector),
  wrong-audio, MCQ permutation, 3 paraphrases, explain-format subset.
- L1/L2/L3: DSP floor / linear probes per layer / behavioral accuracy —
  dissociations localize failures.
- Representation ladder: audio → audio+features → features-only → symbolic,
  ± few-shot.
- Encoder comparison: MERT (music-SSL) vs Whisper (ASR) vs CLAP (contrastive),
  identical probes.
- Generation battery: single-constraint prompts scored by DSP, minus
  unconstrained base rate; vocabulary-vs-theory prompt pairs.

## Results (as of 2026-08-05 — 6 Track A models + Track B probes + Track C–H causal fine-tuning landed; GPT-4o-audio out of scope, no API access)

Full per-model/per-task numbers live in `experiments/results/trackA/analysis_workbook.xlsx`
(sheets `audio`/`no_audio`/`wrong_audio` for summary stats, `data__<model>` for every raw
job) and `experiments/results/trackB/probes/*.csv` + `experiments/results/trackB/analysis/`
(combined CSVs + plots). This section is the distilled version.

### Behavioral battery (Track A) — capability heatmap
6 models scored: Qwen2-Audio-7B, Qwen2.5-Omni-7B, Qwen3-Omni-30B-A3B, Audio-Flamingo-3,
Music-Flamingo-2601, Gemini-2.5-Pro (via Portkey). GPT-4o-audio is out of scope (no
OpenAI API access) — 6 models is the final Track A roster unless that changes.

- **Instrument ID is the one clean, fully-validated task**: large positive audio_gain for
  every model (+16 to +71pp vs no-audio), and near-ceiling probe accuracy (whisper 94%,
  mert 91%). Use this as the sanity-check reference cell for any other task's numbers.
- **Refusal behavior on no-audio controls is not uniform and is diagnostic on its own**:
  Gemini-2.5-Pro refuses 57% of no-audio jobs (correctly flags nothing's there); Qwen2-Audio
  refuses 13%; the other four models (Qwen2.5-Omni, Qwen3-Omni, AF3, Music-Flamingo) refuse
  **0%** — spot-checking their raw responses shows they answer a specific, confident MCQ
  letter with no audio present at all (e.g. AF3/Music-Flamingo pick "4 beats" on essentially
  every no-audio beats_per_bar item). This is hallucination, not graceful degradation, and
  it means their no-audio accuracy numbers reflect pure text-prior guessing.
- **`beats_per_bar` audit (2026-07-31, next action #2 closed)**: manually checked for a
  scoring/parsing bug across all 6 models — there isn't one. `parse_response`/`is_correct`
  apply the same generic open-ended substring match used by every other open-format task
  (no beats_per_bar-specific code path to be broken), and the wrong-audio control follows
  the same fixed 10% sampling fraction (`WRONG_AUDIO_FRACTION` in `jobs.py`) as all 13
  tasks — beats_per_bar isn't singled out by the harness. What IS true: the wrong-audio
  control for this task has only **n=14** items (10% of 126, same design fraction used
  everywhere), and re-deriving the per-model deltas directly from `scored__*.parquet` shows
  beats_per_bar's mean inversion (+0.156 across 7 models, i.e. wrong-audio scores 15.6pp
  *higher* than correct-audio on average) sits at roughly 1 standard deviation of its own
  noise (std 0.141, n=14/model) — directionally consistent (5/7 models positive) but not
  the clean, distinctly-worse-than-every-other-task signal the original framing implied.
  For comparison, `tempo_bpm`'s wrong-audio control (n=2/model) has std 0.35 — much noisier,
  same design. **Revised verdict**: treat beats_per_bar as a *low-confidence, directionally
  suggestive* flag (probably not a genuinely useful audio signal for most models — accuracy
  never clears ~35% in any condition either) rather than "structurally broken" — that
  phrase overclaimed what an n=14 control can support. Don't run a Track-C-style LoRA arm
  on this task without first enlarging the wrong-audio sample (e.g. bump
  `WRONG_AUDIO_FRACTION` for this task alone, or pool multiple paraphrases per stimulus)
  to get a control precise enough to trust.
- **`mode_id` audit (2026-07-31, same pass)**: same check, weaker signal — mean wrong-audio
  delta +0.070 (std 0.169, n=14/model), well within noise, no action needed. No evidence of
  a scoring bug here either.
- **`progression_id` (chord progression naming): 0% open-ended accuracy in 4 of 6 models**
  (Gemini, AF3, Qwen3-Omni, Qwen2-Audio) — MCQ scores on this task are propped up entirely
  by 4-way multiple choice; treat only the open-format number as real.
- **`key_id`'s MCQ score is largely a scaffolding artifact for some models**: Gemini scores
  ~42% MCQ but **0/20 (0%) on open-format** key naming. Music-Flamingo is the exception —
  82% MCQ, 40% open — still a big gap but a real skill underneath, corroborated by it also
  showing the largest wrong-audio-control drop (see below).
- **Wrong-audio control (does accuracy drop when the audio is swapped for an unrelated
  clip, holding the question fixed?)** — the cleanest single "is this model actually
  listening" signal:

  | Model | Δaccuracy, correct→wrong audio |
  |---|---|
  | Music-Flamingo | **+10.8pp** (best evidence of real listening) |
  | Gemini-2.5-Pro | +5.8pp |
  | Qwen2-Audio | +4.0pp |
  | Qwen3-Omni | +3.3pp |
  | Audio-Flamingo-3 | +0.6pp (barely reacts to the swap) |
  | Qwen2.5-Omni | **−1.0pp** (no measurable reliance on correct audio) |

- Cents psychometric thresholds, tuning-detection curve, tempo octave-error histogram:
  not yet formally plotted — accuracy numbers exist in the workbook, curves TODO.
- Harness validation (Qwen2-Audio vs its published MMAU-music/MuChoMusic number): still
  not done — see PROJECT_STATE.md known gaps.

### Explanations analysis (manual, `explain__<model>` sheets / review CSVs)
Spot-checked `beats_per_bar` open + explain responses across all 6 models:
- **Music-Flamingo & Audio-Flamingo-3**: answer "4 beats" almost unconditionally,
  audio or no audio — a text prior, not a measurement.
- **Qwen2.5-Omni & Qwen3-Omni**: "explain" format gives specific timestamps
  (e.g. "Beat 1: 0.00s, Beat 2: 1.00s…") but they're rounded to the nearest 0.5–1s and
  almost always land on a 4-beat answer regardless of ground truth — confident-sounding
  but not actually tracking the audio.
- **Qwen2-Audio**: mixed 3/4 guesses, closest to chance.
- **Gemini**: most varied answers and the best hit rate in spot checks, but still far
  from reliable.

### Representation ladder
Not started — battery v2 (§ PROJECT_STATE decision 11).

### Probes (Track B)
- **Encoder comparison — what's common to MERT and Whisper vs CLAP**: MERT-330M and
  Whisper-enc are both *frame-level* encoders (one vector per ~13–20ms) and both decode
  `beats_per_bar` (76%/96% vs 20% chance), `octave_id` (98%/99% vs 33% chance),
  `tuning_judgment` (90%/87% vs 50% chance) and `instrument_id` (91%/94% vs 25% chance)
  far above chance. CLAP — a *clip-level* contrastive encoder, pooled for caption-matching —
  is markedly weaker on all four. Reading: fine temporal/pitch detail survives frame-level
  pretraining regardless of objective (ASR vs music-SSL) but is destroyed by CLAP's global
  pooling. This is a structural (architecture-of-pretraining) explanation, not a fluke.
- **MERT > Whisper specifically on pitch-precision tasks (H2 support)**: `pitch_note_id`
  (mert 74% vs whisper 45%, chance 8%) and `key_id` (mert 21% vs whisper 12%, chance 4%,
  both weak in absolute terms). Consistent with MERT's CQT-reconstruction pretraining
  target injecting pitch/harmonic inductive bias that Whisper's ASR objective never needed.
- **L2/L3 dissociation** — best encoder probe accuracy vs best behavioral accuracy, per
  task (full table + plot in `experiments/results/trackB/analysis/`):
  - **Alignment-gap candidates (L2 ≫ L3)**: `beats_per_bar` (96% vs 34%), `octave_id`
    (99% vs 53%), `tuning_judgment` (90% vs 58%), `cents_discrimination` (62% vs 43%),
    `note_count` (52% vs 45%) — the L2-high/L3-low pattern H4 predicts; shortlist for
    Track C, **except `beats_per_bar`**, which the wrong-audio evidence above suggests may
    be a broken/prior-driven task rather than a genuine alignment gap — resolve before
    committing a LoRA arm to it.
  - **Consistent (L2 ≈ L3)**: `instrument_id`, `pitch_note_id`.
  - **L3 > L2 (red flag, needs care)**: `key_id` (probe 21% vs Music-Flamingo 75%),
    `interval_id`, `mode_id`, `chord_quality`. Caveat: these probes are on *generic*
    standalone encoders (MERT/Whisper-large-v3/CLAP), not necessarily the exact encoder
    inside the winning LALM (e.g. Music-Flamingo's own AF-Whisper) — re-probe the model's
    own encoder before concluding the behavioral score is priors rather than a genuinely
    different, better internal representation.
- **Key-error structure, done correctly**: the key_id MCQ distractors are *designed* as
  circle-of-fifths neighbors + the relative key (`musicprobe/prompts.py`), so "musically
  structured wrong answers" in MCQ mode are built into the options, not evidence of
  listening — don't use MCQ confusion matrices for this task as a listening signal.
  Redone on open-format-only responses (n=20/model, noisy): structured-error rate
  (fifths-adjacent/relative/parallel-key) ranges 7.7% (Qwen3-Omni, ~chance) to 46.2%
  (Qwen2-Audio) — no clean story yet, sample too small per model to trust beyond flagging
  Qwen3-Omni's errors as closest to pure guessing.
- **Attention diagnostic**: only run for Qwen2-Audio-7B so far (`gpu/attention_audio.py`
  not yet pointed at the other 4 open models — see PROJECT_STATE next actions). Its shape
  (rise to a ~0.4–0.5 peak on audio tokens around layers 3–5, decline to ~0.2–0.28 by the
  final layer) is nearly identical across all 10 tasks tested — inconclusive on
  task-specific attention allocation for this model; needs replication on stronger models
  before drawing any conclusion about attention-to-audio being task-sensitive.
- 12-TET scalloping in probe error: not yet analyzed.

### L1 DSP floor — is the information even recoverable without a model at all?
`musicprobe/l1_baselines.py`, pure numpy/scipy, no learning, no essentia needed for these
10 tasks (extended from 4/13 to 10/13, 2026-08-05). Proves whether the ground-truth label
is physically recoverable from the rendered audio at all — if L1 succeeds and a model
fails, the failure is in the model, not the stimulus.

| task | L1 acc | n | chance | 
|---|---|---|---|
| octave_id | 1.00 | 72 | — |
| cents_discrimination | 1.00 | 180 | — |
| pitch_note_id | 1.00 | 72 | — |
| tempo_bpm | 0.82 | 60 | — |
| key_id | 0.80 | 96 | ~4% (24-way) |
| chord_quality | 0.60 | 96 | 12.5% (8-way) |
| tuning_judgment | 0.63 | 120 | 50% |
| interval_id | 0.54 | 144 | ~8% (12-way) |
| note_count | 0.40 | 100 | ~20–30%ish |
| mode_id | **0.25** | 104 | ~8% (13-way) |

- `octave_id` at a clean 100% is the sharpest confirmation yet that its Track C fix
  (LoRA, +0.50/+0.59) was recovering a real, trivially-present signal — the info was never
  the bottleneck, only the readout was.
- `mode_id` is the weakest signal here (barely 3x chance) — and this now agrees with **two
  other independent methods**: the L2 own-encoder probe (barely above chance for `mode_id`
  specifically, see next section) and L3 behavioral accuracy (~chance, 21–25% on a 4-way
  MCQ). Three methods converging on the same task being the hardest in the battery is a
  much stronger claim than any one of them alone — and it lines up with an existing known
  gap: mode melodies are random diatonic walks, not musician-composed, which may be
  confounding stimulus difficulty with genuine model difficulty.
- `chord_quality`/`interval_id`/`key_id` all sit well above chance (54–80%) despite the
  harmonic cluster showing weak behavioral gains from any front-end tried so far (Track G) —
  this is exactly the "info is there, front-end hasn't found the right shape yet" case, not
  a "nothing to recover" case.
- Not yet covered: `beats_per_bar`, `progression_id` (need real beat/chord-sequence
  tracking — autocorrelation-only isn't trustworthy as a floor claim), `instrument_id`
  (already near-ceiling behaviorally, 94%/91%, so the L1 question isn't the interesting one
  there). Genuinely blocked on essentia/madmom, which won't build on the laptop — needs the
  H100/Linux box.

### Causal fine-tuning (Tracks C–F) — testing the alignment-gap hypothesis (H4)
Track B's L2≫L3 shortlist (`octave_id`, `note_count`, `tuning_judgment`, `cents_discrimination`)
predicted these four tasks were alignment-fixable — info decodable from the encoder but not
reaching the readout. Tracks C–F test that causally, focused on the two hardest cells
(the microtone pair). Full numbers: `experiments/results/trackA/track{c,d,e,f}*`.

- **Track G — chromagram front-end for the harmonic cluster (2026-08-05, both nulls)**:
  tests whether Track D-zoom's "give it a rendered chart" fix generalizes past pitch to
  `key_id`/`mode_id`/`chord_quality`/`interval_id` (never targeted by C–F), via a 12
  pitch-class × time chromagram (the harmonic analogue of the F0-contour). Same paired
  discipline as D/E (mixed-condition training, 3-seed McNemar, wrong-image/wrong-audio
  mechanism controls). Result: **null on every task** — `key_id` Δ=−0.07 (p=.26),
  `mode_id` Δ=+0.04 (p=.57), `chord_quality` Δ=+0.13 (p=.11, largest point estimate but
  n=72, smallest cell), `interval_id` Δ=+0.03 (p=.64); all 95% CIs include 0. Mechanism
  checks: wrong-chromagram ≈ no-chromagram on all 4 tasks (content isn't being read even
  on the rare significant nudge); wrong-audio+correct-chromagram only crashes `key_id`
  (Δ=−0.14, p=.02, so that one task still leans on audio) — the other three are
  inconclusive either way. Unlike pitch, the harmonic cluster doesn't respond to an
  explicit-chart front-end; two of the four tasks (`mode_id`, `chord_quality`) are also
  sitting near chance already, so floor effects may be part of the story.
- **Track H — in-audio reference tone for `tuning_judgment` (2026-08-05, null)**: tests
  whether Track D-zoom's reference-line fix for absolute tuning survives being delivered
  in-AUDIO (a second tone mixed into the clip) instead of switching to a visual chart.
  Result: flat null — `reftone` vs `plain` Δ=+0.01 (p=1.0); critically, `wrong_reftone`
  (reference mistuned ±1–4 semitones) doesn't mislead the model any more than a *correct*
  reference does (Δ=+0.02 vs plain, p=.82) — both land at ~0.53, statistically
  indistinguishable from no reference at all. The model isn't comparing target-to-reference
  in audio; it's simply not reacting to a second tone being present, right or wrong. Confirms
  the D-zoom reference-line effect is specifically a *visual* trick (reading an annotated
  position on a chart), not a general "give it something to compare against" principle —
  reinforces the project's running "can't hear microtones" finding via a channel where
  comparison should in principle be easiest.
- **Track F-aug, leakage bug fixed (2026-08-05)**: the original aug run (`0aed136`) sampled
  training pitches from a range overlapping the held-out eval band; `generate_aug.py` now
  caps training pitches (`MAX_TRAIN_MIDI = 70`, a 1-semitone margin). Corrected audio-only
  accuracy on held-out (never-trained) pitches: cents 0.62 (small data) → 0.78 (first clean
  pass) → **0.74** (final clean, all 3 seeds) vs 0.89 in the leaky run — about half the
  leaky jump was real; more clean augmented data does genuinely raise relative-pitch
  discrimination. Tuning: 0.51 → 0.68 → **0.62** vs 0.83 leaky — nearly halved once the
  leak was fixed, now sitting close to the 2-way forced-choice majority rate, i.e. **not**
  convincing evidence of learned absolute-tuning perception (consistent with Track H
  above — absolute tuning stays unlearnable via this pathway). The fusion-null verdict
  itself is unchanged: Δ still ns, `image+wrong_audio` still craters, model still ignores
  the fused stream and relies on audio.

- **Track C — AF3, 3-arm LoRA (`llm_only` / `llm_encoder` / `control`), matched-baseline
  delta**: `octave_id` (+0.50 / +0.59) and `note_count` (+0.43 / +0.40) confirm the
  prediction — real, causally-recovered readout-alignment gaps. `tuning_judgment`
  (+0.08 / +0.08) and `cents_discrimination` (−0.07 / +0.04) do **not** — both resist
  fine-tuning even when the encoder itself is also tuned, meaning microtone info either
  isn't in AF3's representation or isn't recoverable through this pathway. The H4
  prediction was half right: it holds for coarse/readout-level pitch info, not for
  microtone perception.
- **Track D, three iterations (Qwen2.5-Omni + a visual pitch aid), same paired within-model
  eval (3 seeds, McNemar exact test, wrong-image/wrong-audio controls) throughout**:
  1. *Phase 1* (single LoRA arm, spectrogram, trained on `image` condition only): looked
     like a win (cents 0.67→0.77) but the `no_image` eval was out-of-distribution for a
     model only ever trained with an image — confounded, not trustworthy.
  2. *Conclusive* (mixed-condition training so both `image`/`no_image` are in-distribution):
     a precise null — Δacc 95% CI includes 0 on every task (cents [-0.05,+0.05],
     tuning [-0.13,+0.16], octave [-0.11,+0.03], note_count [-0.10,+0.15]). Mechanism
     controls confirm the model **ignores the spectrogram entirely**: wrong image ≈ correct
     image; wrong audio + correct image collapses to chance anyway.
  3. *Force* (same spectrogram + modality-dropout training, so the image becomes
     necessary): succeeds at the mechanical goal — wrong image now craters performance
     (cents p=.003, tuning p=.016) — but accuracy still doesn't improve (cents Δ+0.05 ns).
     The bottleneck moved from "model ignores the image" to "the image doesn't contain
     the fine-grained info" — a fixed-scale spectrogram can't resolve 5¢ (~0.4px).
  4. *Zoom* (a zoomed F0-contour chart with an in-tune reference line, `f0_contour.f0_zoom_path`,
     same modality-dropout training): **fixes both** — cents 0.55→0.94 (Δ+0.39, p<1e-4),
     tuning 0.53→0.89 (Δ+0.36, p<1e-4). This is the only method across Tracks C–F that
     recovers absolute tuning; the reference line is load-bearing (nothing else tested —
     probe, LoRA, spectrogram, or plain F0 — recovered absolute tuning).
- **Track E — pitch-tracker output as plain text in the prompt** (audio-only, no image,
  same modality-dropout + paired eval): fixes relative pitch (cents 0.62→0.92, p<1e-4;
  octave 0.71→0.84, p=.02) without any visual front-end — the cheaper, more deployable
  option. Tuning stays flat (0.58→0.60, ns): text carries the pitch value but no
  *reference*, so there's nothing to judge "in tune" against — the zoomed image's
  reference line is the specific ingredient tuning needs.
  Mechanism check (both D-zoom and E): wrong feature craters accuracy (so the model reads
  it) but feature+wrong-audio ≈ feature+correct-audio — this is **substitution, not
  hearing**: the external pitch tracker does the perception, the LM only reads and
  reasons over its output. Consistent with Track C's `llm_encoder` null (encoder LoRA
  can't recover what the encoder never captured).
- **Track F — end-to-end learned fusion**: a small trainable MLP projects a frame-level F0
  contour and injects it into the LM's embedding space via unused special-token slots
  (architecturally the "no cheating, no external tracker in the prompt" version of D/E).
  Verified the injection is real — it shifts next-token logits (‖Δ‖=1.78) and the shift
  tracks the pitch value — but behaviorally it's a **null** (cents 0.62→0.64 ns; tuning
  0.51→0.46 ns; wrong-pitch ≈ correct-pitch, i.e. the fused stream is largely ignored).
  Reading: routing pitch through an interface the model already understands (numbers as
  text, or a chart) works; asking it to learn a *new* embedding-space interface from
  ~348 examples does not — this reads as a data-scale limit on the learned-fusion
  approach, not evidence the information is unusable in principle.

### Generation models (MusicGen-medium, 342 clips, 6 constraint families)
- **Tempo** (96 clips): exact-BPM match only 40%; loosened to "half/double tempo also
  counts" (octave-equivalent), 69%. Model systematically drifts toward its own preferred
  tempo range rather than the requested BPM, worst at the high end (asked 158 BPM,
  averaged 113).
- **Key** (72 clips): only 31% landed in the requested key.
- **Register** (24 of 36 clips got a pitch reading at all): **0% satisfied** — asked to
  stay above C5, it generated a full octave-plus lower every time; asked to stay below C3,
  the few that extracted a pitch overshot it. Register/tessitura instructions are
  essentially ignored.
- **Meter & mode**: deliberately not auto-scored (`genmodel/score_gen.py` — needs downbeat
  tracking / manual verification, by design, not a bug). WAVs are gitignored; need
  Drive/scp from the H100 box to listen.

## Conclusions (partial — Tracks C–H causal fine-tuning now landed 2026-07-26 to 08-05; GPT-4o-audio out of scope)
- **Per-skill verdict so far**:
  - *Alignment-fixable, causally confirmed (Track C)*: `octave_id` (+0.50/+0.59 over
    baseline), `note_count` (+0.43/+0.40) — LoRA on the existing audio→LM pathway recovers
    these, exactly as the L2≫L3 pattern predicted.
  - *NOT alignment-fixable by LoRA alone, but fixable by an engineered front-end
    (Tracks C→D→E)*: `cents_discrimination` and `tuning_judgment` resisted fine-tuning in
    Track C even with the encoder tuned — the H4 "just LoRA it" prediction was wrong for
    these two. What worked instead: pitch-tracker-as-text fixes relative pitch (cents,
    Track E) with no image needed; a zoomed F0-contour chart with an in-tune reference line
    fixes both relative and absolute tuning (Track D-zoom) — the reference line is the
    specific ingredient absolute tuning needs, and nothing else tried (probe, LoRA,
    spectrogram, raw learned fusion) recovered it. Mechanistically this is substitution
    (external tracker perceives, LM reads/reasons), not the model learning to hear
    microtones — a real fix for a deployed system, not evidence of a closed perception gap.
  - *Reference-line fix does NOT generalize past pitch, and does NOT survive a channel
    switch (Tracks G/H)*: a chromagram front-end for the harmonic cluster
    (`key_id`/`mode_id`/`chord_quality`/`interval_id`) is a clean null on all four tasks —
    the explicit-chart trick that rescued pitch doesn't transfer to harmony. Delivering the
    same "reference to compare against" idea in-audio instead of visually (Track H,
    `tuning_judgment`) is also a clean null, and the mechanism control shows the model
    isn't even attempting a target-vs-reference comparison in audio (wrong reference ≈
    correct reference). Together these narrow the D-zoom finding to something quite
    specific: an *annotated visual position*, for *pitch specifically* — not a general
    "give the model a reference" or "give the model a chart" principle.
  - *Learned end-to-end fusion does not work at this data scale (Track F)*: injecting a raw
    F0 feature into embedding space via a trainable adapter is verifiably ignored
    behaviorally (~348 examples isn't enough to learn a new modality interface from
    scratch), even though the same information fixes the task when delivered through an
    interface (text, image) the model was already pretrained to read. General lesson for
    any future modality-injection experiment in this project: prefer reusing an existing
    interface over learned raw fusion, unless training data scales up substantially.
  - *Needs a task-validity check before any LoRA arm*: `beats_per_bar` — probe says the
    info is there, but the wrong-audio control says the model isn't using audio at all (in
    either direction), which reads more like a broken task/prompt than an alignment gap.
    Track C ran without it for exactly this reason; still unaudited (PROJECT_STATE next
    actions #2).
  - *Needs re-probing the model's own encoder before verdict*: `key_id`, `mode_id`,
    `chord_quality`, `interval_id` (L3 > generic-encoder L2) — re-probed on each model's own
    encoder (2026-07-24); accuracy stayed modest, not clearly beating generic MERT/Whisper/
    CLAP baselines, so behavioral success on these four is more likely priors than a richer
    internal representation. No Track-C-style causal test run on this group yet.
  - *Data-fixable / genuinely weak across the board*: `progression_id`, `mode_id`,
    `interval_id`, `chord_quality` (all near floor on open-format).
- **Representation recommendation (first concrete data point, pitch only)**: for a model
  we fine-tune or design ourselves, relative pitch should be exposed via a compact
  in-context symbolic/text feature (cheap, no rendering needed, Track E); absolute tuning
  needs an explicit *reference* alongside the pitch value, not just the value itself
  (Track D-zoom) — a scale-anchored representation, not a bigger/sharper image, is what
  was missing. Full representation-ladder verdict across all tasks still pending battery v2
  (PROJECT_STATE next action #6), which Tracks D-zoom/E's harness can now seed rather than
  building from scratch.
- **Implications for training our own model**: two findings so far, both pitch-specific but
  likely to generalize: (1) frame-level (non-clip-pooled) encoders beat clip-level/pooled
  ones regardless of ASR-vs-music-SSL pretraining objective (MERT/Whisper vs CLAP, Track B);
  (2) when perception genuinely isn't reaching the readout, prefer an engineered front-end
  that reuses a pretrained interface (numbers-as-text, referenced charts) over a raw
  learned-fusion adapter — the latter needs training-data scale this project doesn't have
  (Track F). Still too early for a full architecture recommendation; both points should be
  re-checked against the representation ladder and any non-pitch tasks that turn out to have
  the same L2≫L3 profile.
