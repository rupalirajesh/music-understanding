# The Question Battery — tier list, data sources, and what each result means

This is the concrete version of the "tier list of questions in increasing complexity"
task. Every row is implemented (✅) or specced (📋). Tasks are ordered so that each
tier's questions *presuppose* the tier below: a model that can't hear pitch classes
(T1) cannot honestly identify keys (T2), so a key-ID success without pitch-ID success
is itself evidence of shortcutting.

## How to read a result (the diagnostic logic)

For every task we get: accuracy with audio, accuracy with **no audio** (text-prior
control), accuracy with **wrong audio**, the **confusion matrix**, and — for open
models — linear-probe accuracy at each layer (L2). The interpretation table:

| Pattern | Verdict |
|---|---|
| audio ≈ no-audio accuracy | task never measured hearing → text priors (MuChoMusic failure mode) |
| audio ≫ no-audio, errors musically structured (fifths-adjacent keys, tempo octaves, bend↔slide) | genuine but imprecise perception |
| audio ≫ no-audio, errors uniform | weak signal + guessing among options |
| L2 probe high, behavior low | **alignment gap** → cheaply fixable by fine-tuning (Track C) |
| L2 probe low everywhere, L1 fine | encoder/architecture problem → NOT fixable with data |

## Tier 1 — Atomic perception (all synthetic, perfect ground truth) ✅ implemented

| # | Task | Question | Stimuli | N (full) | Why it's diagnostic |
|---|---|---|---|---|---|
| 1.1 | `pitch_note_id` | Which note is this (pitch class)? | 1 note × 12 pc × 3 octaves × instruments × soundfonts | 144 | The floor. Absolute pitch from audio; distractors include ±1 semitone. |
| 1.2 | `octave_id` | Which octave? | same audio as 1.1 | 144 | Octave errors ≠ pitch-class errors; separates register coding from chroma coding. |
| 1.3 | `interval_id` | Name the interval | 12 intervals × melodic asc/desc + harmonic, random roots | 144 | Relative pitch. Random roots kill absolute-pitch shortcuts. Harmonic-vs-melodic gap ⇒ simultaneity is the bottleneck. |
| 1.4 | `cents_discrimination` | 2nd tone higher/lower/same? Δ∈{5,10,25,50,100}¢ | numpy tone pairs, off-grid base pitch | 180 | **Psychometric curve** → the model's pitch-resolution limit in cents. Tests H2 (threshold ≥50¢). Humans (trained): 5–10¢. |
| 1.5 | `tempo_bpm` | BPM? (open-ended number) | click tracks, BPM continuous 55–185 | 60 | Error histogram: peaks at ±1 log2 = honest octave errors; spikes at 120 regardless of truth = prior. |
| 1.6 | `beats_per_bar` | How many beats per cycle? | accent-patterned clicks (3,4,5,6,7 beats) | 100 | 5 and 7 are the money cells: rare in training data. NOTE: audio cannot convey the notated denominator (4/4 vs 4/8 are identical sounds), so we ask the audible question. A 3/4-vs-6/8 *grouping* task (3×2 vs 2×3) is specced for later with real drum patterns. |
| 1.7 | `note_count` | How many simultaneous notes? | 1–5 note clusters | 100 | Polyphony (= multiple notes sounding at once) resolution — prerequisite for chord tasks. |
| 1.8 | `tuning_judgment` | Is this note in tune? | tones detuned 0–50¢ off the semitone grid | 120 | **The 12-TET grid probe, behavioral half.** A model whose internal pitch snaps to Western semitones loses the evidence and calls everything in tune → flat curve. Same stimuli carry exact fractional pitch for the Track-B probe (scalloped probe error = quantized representation). |

## Tier 2 — Musical structure (synthetic, perfect ground truth) ✅ implemented

| # | Task | Question | Stimuli | N | Why it's diagnostic |
|---|---|---|---|---|---|
| 2.1 | `key_id` | What key? | 24 keys × {scale run, I-IV-V-I progression + melody} | 96 | MCQ distractors = fifths neighbors + relative key, so the confusion matrix separates listening errors from guessing. Scale vs progression form: pitch-set enumeration vs true tonality. (Terminology note: the progression form is a tonality-establishing chord sequence, not a cadence-type task — classifying authentic/plagal/half/deceptive cadences is a good future Tier-2 task, specced below.) |
| 2.2 | `mode_id` | What scale/mode? | 13 modes × {bare scale, melody-over-drone} | 104 | Drone pins the tonic → isolates mode quality from tonic-finding. Dorian/mixolydian confusions are informative; blues/pentatonic are the "non-classical" cells. |
| 2.3 | `chord_quality` | Chord type? | 8 qualities × {block, arpeggiated}, inversions as factor | 96 | Arpeggiated-succeeds/block-fails ⇒ simultaneity bottleneck (links to 1.3, 1.7). Inversions test whether "minor" is a root-position template or a concept. |
| 2.4 | `progression_id` | Which progression? | I-IV-V-I, I-V-vi-IV, ii-V-I, 12-bar blues × 8 keys | 32 | Requires chord tracking *over time* + relative harmonic labels. 12-bar blues also has a strong text prior — watch the no-audio control. |
| 2.5 📋 | `modulation_detect` | Does the key change? Where? | progression, ±2 semitone shift halfway vs none | ~60 | Change detection is easier than naming — if this fails too, harmony tracking is absent, not just unverbalized. |
| 2.6 📋 | `cadence_type` | Authentic / plagal / half / deceptive? | V-I, IV-I, x-V, V-vi endings in context | ~80 | Functional harmony hearing — a step past chord ID toward what the chords *do*. |
| 2.7 📋 | `grouping_3v6` | 3/4 or 6/8 feel? | drum patterns, 3×2 vs 2×3 grouping at matched cycle length | ~60 | Same cycle length, different grouping — pure metrical-hierarchy perception. |

## Tier 3 — Techniques, semantics, real audio 📋 specced (Phase 2)

Ground truth comes from labeled datasets, not synthesis. Contamination risk is real
(these datasets are old and public) — flag every result with a contamination note,
and prefer held-out/fresh recordings for headline claims.

| # | Task | Data source | Why |
|---|---|---|---|
| 3.1 | Vocal technique ID (vibrato, belt, breathy, fry, trill…) | **VocalSet** (17 techniques × 20 singers; split by singer) | Fast, fine pitch+timbre events near the token-rate limit — H5's behavioral face. |
| 3.2 | Guitar technique ID (bend, slide, hammer-on, palm mute, harmonics) | **GuitarSet**, IDMT-SMT-Guitar | bend↔slide = honest confusion; bend↔palm-mute = not listening. |
| 3.3 | Instrument ID in isolation / in mix | NSynth / MedleyDB stems | Timbre is models' strength — this anchors the "what they CAN do" side. |
| 3.4 | Genre + subgenre, **with cue ablation** | MTG-Jamendo; same piece re-rendered MIDI→soundfont | The H3 test: does genre survive when production texture is stripped and only composition remains? |
| 3.5 | Tempo/beat on real mixes | Harmonix Set / GTZAN-rhythm (caution: contamination) | Bridges 1.5 to ecological validity. |
| 3.6 | Structure (verse/chorus, where's the drop) | SALAMI, long clips | Long-context degradation; needs Gemini-class context. |
| 3.7 | Symbolic contrast arm | same T1/T2 questions as ABC notation text | The "knows theory, can't hear" calibration — runs on Claude too (no audio input needed). Directly separates knowledge from perception. |

## Cross-cutting factors baked into every synthetic stimulus

- **instrument** (GM program: piano/violin/flute/synth) and **soundfont** (3 fonts) —
  always recorded → any probe/eval can hold out a timbre (leakage guard) and any
  failure can be sliced by timbre ("can it name notes on piano but not violin?").
- **register/root randomization** — kills absolute-position shortcuts.
- **seed** per stimulus — full reproducibility.

## Controls (attached automatically by the jobs builder)

1. **No-audio** on 30% of stimuli — the text-prior detector. *The single most
   important number in the study is `audio_gain = acc_audio − acc_no_audio`.*
1b. **Explain jobs** on 15% — the model answers, then must describe what it
   heard with notes/timestamps. Not auto-scored (deliberately: this is the
   manual-analysis file, `explanations.csv` in the review export). A correct,
   specific explanation is strong evidence of listening; but treat it as
   evidence, not proof — models can confabulate fluent rationales around a
   guessed answer, so weigh *specific verifiable details* (correct timestamps,
   correct note names beyond the answer itself), not eloquence.
2. **Wrong-audio** on 10% — prompt-leakage detector.
3. **MCQ position permutation** — every item independently shuffled.
4. **≥3 paraphrases** per task, sampled per item.
5. **MCQ + open-ended** dual format on 25% — MCQ inflation check.
6. **Loudness normalization** — every stimulus peak-normalized to −3 dBFS.

## Sample size (the question you asked)

Rules of thumb used for the Ns above, all from the binomial CI / two-proportion
power formula:

- **Detecting "above chance" (25% MCQ)**: ~40 items already gives 95% power to
  detect true accuracy ≥50%. Easy.
- **Pinning one accuracy to ±7pp (95% CI)**: n ≈ 100 per task-condition cell. This
  is the budget most tasks are sized to.
- **Comparing two models / two conditions** and calling a 10pp difference real:
  n ≈ 350–400 *per arm* (80% power). We do NOT size every cell for this — instead,
  claims about differences ride on (a) many tasks moving the same direction
  (sign-test logic across the battery beats a big-n single task), and (b) paired
  design: identical stimuli across models makes McNemar's test applicable, which
  roughly halves the n needed vs independent samples.
- **Psychometric curves**: ≥30 trials per difficulty level (we use 30/Δ) is the
  standard psychophysics floor for a stable threshold fit.
- **Factor slices** (e.g. "fails on violin only"): treat as hypothesis-generating
  unless the slice has ≥50 items; promote interesting slices to a targeted
  follow-up generation run (synthesis makes this free — regenerate 200 more of
  exactly that cell).

Bottom line: **~1,100 synthetic stimuli (implemented) ≈ 1,500–2,000 API calls per
model** including controls; at Tier-1/2 clip lengths that's cheap even on paid APIs,
and per-task conclusions come with ±7–10pp CIs, which is enough to separate
"can't do at all" (≈chance) from "does imperfectly" (50–80%) from "solved" (>90%)
— the resolution RQ1 actually needs.
