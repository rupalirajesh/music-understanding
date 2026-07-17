# How much do music GENERATION models understand?

The call raised two claims worth testing rather than assuming:

1. *"Generation models have no music understanding really — they play from feel."*
2. *"If we gave them understanding, they could do 'generate in 5/4' / 'raag bhairav' /
   'only 5th octave'."*

Both become measurable with one trick: **a generation model's "understanding" is its
constraint-adherence, and constraint-adherence is scored by our L1 stack.** We don't
need the model's internals — we ask it to satisfy a verifiable musical constraint and
run classical MIR on the output. This is exactly the mirror image of Track A: for
understanding models we fix the audio and grade the words; for generation models we
fix the words and grade the audio.

## The battery (mirrors TASKS.md tiers)

| # | Prompt constraint | Verifier (L1) | Understanding-model twin |
|---|---|---|---|
| G1 | "at exactly 97 BPM" (sample BPM continuously — never 120) | beat-track → tempo | `tempo_bpm` |
| G2 | "in 5/4 time" / "in 7/8" vs "a waltz" | beat + downbeat tracking | `meter_id` |
| G3 | "in F# minor" / "in D Dorian" | key/chroma detection | `key_id` / `mode_id` |
| G4 | "melody stays above C5" / "only in the 5th octave" | f0 tracking (crepe) | `octave_id` |
| G5 | "solo piano, no other instruments" | instrument tagging (CLAP zero-shot) | instrument ID |
| G6 | "12-bar blues progression" | chord estimation → progression match | `progression_id` |
| G7 | "ends on the tonic" / "hits a G5 at the end" | f0 of final phrase | pitch ID |

Each cell: ≥20 generations per constraint value, constraint values randomized
(97 BPM, 83 BPM… not round numbers), plus the **prior control**: generate with NO
constraint and measure the base rate (if unconstrained output is 4/4 at 120 BPM
half the time, "can do 4/4" means nothing — exactly like our no-audio control).

Score = P(output satisfies constraint) − P(unconstrained output satisfies it).
The result is a *capability heatmap for generation models directly comparable to the
understanding heatmap* — same musical properties, same L1 verifiers, same controls.
Your Suno observation ("waltz works, 5/4 doesn't") becomes a quantified row: it
predicts G2-waltz high, G2-5/4 ≈ base rate — i.e. adherence tracks *training-data
vocabulary density*, not music theory. If that pattern holds across the battery
(works ⇔ describable in caption-speak), that's strong evidence for the "feel, not
understanding" claim, and it localizes WHY: text-conditioning vocabulary is the
bottleneck, the same caption-granularity ceiling that limits understanding models
(RESEARCH_PLAN §1.4).

## Models to test

- **MusicGen (small/medium/melody)** — open weights, runs on Colab; text-conditioned;
  the primary target since it's also the most realistic *fine-tuning* target if we
  end up training a generation model. Also supports melody conditioning → tests
  "keep this melody but change key" style instructions.
- **Stable Audio Open** — open weights, different conditioning mechanism (timing +
  text), good contrast.
- **Suno / Udio** — no API for controlled runs; do a small manual batch (10–20
  prompts each) for the paper-trail comparison, not for statistics.

## The deeper probe (Track B for generation models — later)

MusicGen's LM operates on EnCodec tokens. The same linear-probe machinery from
Track B applies: feed our *synthetic stimuli* through MusicGen's encoder/LM, probe
hidden states for key/tempo/chord. If key is decodable from MusicGen's states but
prompt adherence to "in F# minor" fails → the generation model has an **alignment
gap too** (it "knows" the key of what it's playing but its text interface can't
steer it). That would directly support the combined generation+understanding thesis
from the call: the two capabilities plausibly share representations, and
understanding-style supervision could unlock controllability. This is the
experiment that answers "can the generation model train the understanding model
(or vice versa)?" with evidence instead of vibes.

## Design rules (added after 2026-07-17 discussion)

1. **One constraint per prompt.** Never "97 BPM in F minor" — a failure tells
   you nothing about which constraint broke. Compound constraints are a later
   battery, run only on constraints the model passes individually.
2. **Vocabulary-vs-theory pairs** (the Suno waltz observation, made controlled):
   the same constraint phrased as a style word vs a theory term — "a waltz" vs
   "in 3/4 time", "a march" vs "in 4/4 time". If adherence follows the phrasing
   rather than the constraint, the model learned caption vocabulary, not meter.
   That IS the "data issue" diagnosis: 3/4-ness exists in its training audio in
   abundance (waltzes!), but the string "3/4" was never linked to it — a
   text-interface gap, not a musical inability. The fix that suggests is
   captioning/instruction data, not architecture.
3. **12-TET snapping via output waveform**: generation models DO give us a
   waveform out (understanding models don't — they only emit text). Pitch-track
   generated solo melodies: do the notes land on the 12-TET grid even when we
   ask for bends/quarter-tones/vocal slides? Distribution of f0 relative to the
   nearest semitone = the output-side version of the representation-grid probe.

## Status

✅ Implemented: `run_musicgen.py` (generation battery: tempo, meter incl.
vocab/theory pairs, key, mode, register + unconstrained baselines) and
`score_gen.py` (adherence scoring via the same L1 verifiers as the
understanding battery; tempo/key/register automatic, beats-per-bar by ear —
lightweight downbeat tracking on generated audio isn't trustworthy).
Runs on the H100 box; not yet executed. Suno/Udio manual batch: TBD.
