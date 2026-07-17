# Archived: Carnatic/Indian-classical module design notes (from plan v1, 2026-07-15)

Deferred per decision to focus on general/Western music understanding first. Everything here
plugs into the L1/L2/L3 + Track A/B/C machinery in RESEARCH_PLAN.md when reactivated.

## Task battery (Indian classical specific)
- Tonic (sruti) identification — foundational: raga is tonic-relative.
- Svara identification given tonic (direct analog of absolute-note ID, comparable cell).
- Raga classification: 5-way MCQ among acoustically distinct ragas → 40-way; separately test
  **allied-raga pairs** (same scale, different characteristic phrases/gamakas, e.g.
  Bhairavi vs Mukhari) — isolates melodic-grammar understanding from scale-content matching.
- Gamaka detection and naming (kampita, jaru, …); presence/absence on matched clips.
- Tala: identify adi/rupaka/misra chapu; beats per cycle; locate the sama.
- Section identification: alapana vs kriti vs tani avartanam.
- Melakarta reasoning as text-only control: does the model know raga theory as text even if
  it can't hear it? (separates knowledge gap from perception gap).

## Key controlled contrasts
| Contrast | Isolates |
|---|---|
| Same raga phrase on piano vs veena vs synth | timbre familiarity vs melodic grammar |
| Dorian melody (Western instr.) vs equivalent raga scale (Western instr.) | cultural labeling vs pitch-set perception |
| Microtonal discrimination in blues-bend vs gamaka context | continuous-pitch ability vs cultural framing |
| Scale-identical allied ragas | phrase-level grammar |

## Data & tools
- **Saraga** via `mirdata`: Carnatic (~250+ recordings) + Hindustani (108 recordings,
  61 ragas); tonic, pitch tracks, sama/tala, sections, phrase annotations.
- CompMusic Carnatic corpus (Dunya), Carnatic Varnam dataset (aligned notation),
  Indian Art Music Raga Recognition dataset, RagaDhvani (2025), Mridangam stroke dataset,
  Carnatic Music Rhythm dataset.
- `essentia` has Indian-music extractors (tonic ID, pitch salience).
- L1 is solved for raga ID: supervised CNNs ≈88% on full CompMusic Carnatic corpus,
  97–99% on 10-raga subsets — so any LALM failure is L2/L3, not signal recoverability.

## Track C variant
Carnatic instruction set (~5–20k QA triples) generated programmatically from Saraga
annotations (tonic QA, svara QA, raga MCQ, gamaka yes/no, tala QA); three-arm LoRA as in the
main plan; hold out ragas AND artists (Saraga has few artists — leakage risk).

## Hypotheses parked with the module
- Raga-as-scale partly encoder-recoverable but behaviorally poor (alignment gap);
  allied-raga discrimination fails at L2 too (representation gap).
- Text-only raga theory knowledge decent in frontier text LLMs → gap is perceptual, not
  knowledge.
- Gamaka perception fails for encoder/token-rate reasons *before* culture enters —
  testable now via the microtonal psychometrics + temporal-resolution analyses already in
  the main plan.
