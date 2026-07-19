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

## Results (fill as runs land)

### Behavioral battery (Track A) — capability heatmap
| task | Qwen2-Audio | Gemini | GPT-4o-audio | AF3 | Music Flamingo |
|---|---|---|---|---|---|
| (12 tasks × acc / audio_gain) | | | | | |

- Cents psychometric thresholds: ___ (humans: 5–10¢)
- Tuning-detection curve (12-TET snap test): ___
- Tempo error histogram (octave errors vs. 120-prior): ___
- Confusion structure notes (fifths-adjacent keys? m3/M3?): ___
- Harness validation: Qwen2-Audio published ___ vs ours ___

### Explanations analysis (manual)
- listening-vs-guessing verdicts per task: ___

### Representation ladder
- Which rung recovers which task: ___

### Probes (Track B)
- L2/L3 dissociation table: ___
- Encoder comparison (what MERT keeps that Whisper drops): ___
- 12-TET scalloping in probe error: ___

### Generation models
- Adherence lift per constraint: ___
- waltz vs 3/4 (vocab vs theory): ___
- Output-pitch 12-TET snapping: ___

## Conclusions (empty until results)
- Per-skill verdict: data-fixable / alignment-fixable / architectural: ___
- Representation recommendation: ___
- Implications for training our own model: ___
