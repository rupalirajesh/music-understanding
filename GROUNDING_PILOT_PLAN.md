# Grounding pilot — plan draft (2026-08-21, for review before any building)

Per `WHAT_TO_DO.md`: this is a plan and sample items only. Nothing here is
built yet. Do not treat any of this as implemented until it's explicitly
approved and moved into code.

## What this is

The Suite C protocol from `DIRECTION_DECISION.md`: force every analytical
answer to separate three sources of information and label which one produced
each part of the claim. This is the actual novel contribution — not another
model-vs-benchmark accuracy number, but a test of whether an assistant can be
honest about *how* it knows what it's claiming.

Already built and usable as-is:
- **Suite A (retrieved knowledge)**: `benchmark/tier1_pilot.json` /
  `tier2_pilot.json` — 5 songs, ~27 citation-gated facts, already rigorous.
- **Suite B (audio-only counterfactual)**: the existing synthetic battery +
  wrong_audio/wrong_image controls already does this in spirit, just not
  labeled as "Suite B."

What's missing and is the actual subject of this plan: **Suite C** — audio +
a deterministic tool report, scored on the evidence schema, not just the
final answer.

## The evidence schema

Every answer must be structured as:

```
claim:                    <the analytical statement>
evidence_span_seconds:    <timestamp/range in the audio this claim rests on, or "n/a">
measurement_or_tool_output: <the specific tool/DSP output used, verbatim, or "none used">
theory_rule_or_source_id: <the music-theory rule applied, OR a citation ID from Suite A>
confidence:               <stated confidence, 0-1 or low/med/high>
```

A response is only credited as "grounded" if the evidence field names a
*real, checkable* observation — not a restated claim, not a plausible-sounding
but unverifiable assertion (see the Moonlight Sonata case below for exactly
the failure mode this is designed to catch).

## Worked example 1 — using tonight's real result

This is the Moonlight Sonata response we actually collected tonight
(`results/external_benchmarks/real_recordings_qwen25omni.json`), shown against
what a schema-compliant answer would look like.

**What the base model actually said (ungrounded, tonight, real):**
> "This piece is in C# minor. The key signature indicates the presence of
> seven sharps, which is characteristic of C# minor."

Two problems: (1) "key signature" is a notation concept, not something
audible — invoking it from audio alone is a category error; (2) it's also
factually wrong on its own terms — C# minor's actual key signature is 4
sharps, not seven. The right answer, wrong and self-contradictory reasoning.

**What a schema-compliant answer would require:**
```
claim: C# minor
evidence_span_seconds: 0.0-30.0 (full provided clip)
measurement_or_tool_output: L1 key_estimate (Krumhansl correlation) = C# minor, correlation 0.81
theory_rule_or_source_id: n/a (this is a direct measurement, not a theory inference)
confidence: 0.75 (L1 estimator is weak on minor-key pieces per known_gaps in PROJECT_STATE.md)
```
If the model can't produce a real tool output, the honest schema-compliant
answer is "measurement_or_tool_output: none used" and confidence should drop
accordingly — not a confabulated substitute.

## Phrasing convention (decided 2026-08-21)

Every item's question is written in natural, product-like language — how an
actual user would ask, not exam-style ("identify the modulation and its pivot
point"). This is a testing/prompt-design decision, not a training one — it
applies uniformly to this pilot and to any later expansion of it, since it's
a fixed property of how items get written, independent of scale.

| Phenomenon | Natural-language question |
|---|---|
| Pitch/motif transposition | "Is this later phrase the same melody as before, just shifted higher or lower — or is something about it actually different?" |
| Syncopation vs. straight rhythm | "Does this rhythm feel like it's pushing against the beat, or is it landing squarely on it?" |
| Tempo/metric change | "Does the pulse change partway through this clip, or does it stay steady the whole way?" |
| Harmonic function/modulation | "Does this song change key partway through? If so, where, and what's it moving to?" |

## Worked example 2 — a constructed counterfactual (harmonic function)

Phenomenon: **local modulation**. User-facing question: *"Does this song
change key partway through? If so, where, and what's it moving to?"*

- **Base item**: a synthetic 16-bar phrase modulating from C major to G major
  at bar 9 via a secondary dominant (D7).
- **Swapped-target counterfactual**: identical audio up to bar 8, then
  modulates to A minor instead (relative minor, a harder/closer confusion)
  from bar 9 onward — built via the same deterministic synthesis pipeline
  already used for the whole battery, so ground truth is exact by
  construction, no musician needed.
- **Tool report given**: a chord-by-bar list from the L1 chord_quality
  estimator (`bar1: C, bar2: Am, ... bar9: D7, bar10: G, ...`), the deterministic,
  checkable "measurement" the model must cite from.

**Required schema output:**
```
claim: modulates from C major to G major at bar 9
evidence_span_seconds: 14.2-15.8 (bar 9's audio span)
measurement_or_tool_output: chord_quality tool report, bar9=D7, bar10=G
theory_rule_or_source_id: D7 functions as V7/V (secondary dominant), resolving to the new tonic G
confidence: high (tool report and theory rule agree)
```

**Ablations to run on this same item** (per `DIRECTION_DECISION.md`'s Suite C
design): swap the audio only (should change the claim), swap the tool report
only (should change the claim even if audio is correct — tests whether the
model actually reads the tool report or just restates a prior), swap both to
mismatched values (should surface a contradiction, not silently pick one).

## Phenomena for the pilot (4, per `WHAT_TO_DO.md`)

1. **Pitch/motif transposition** — same melodic contour, transposed vs. one
   altered interval.
2. **Syncopation vs. straight rhythm** — onset shifted off/on the beat, tempo
   unchanged.
3. **Tempo/metric change** — BPM change vs. 3/4-vs-6/8 grouping change.
4. **Harmonic function/modulation** — worked example above.

Each gets ~6 items (24 total), all 5 conditions (audio absent / correct audio
/ swapped audio / audio+tool report / audio+swapped tool report) — matching
`WHAT_TO_DO.md`'s exact spec.

## Tools: 3, one per orthogonal facet (decided 2026-08-21)

Not 2 (pitch + time) — **pitch, rhythm, and harmony each get their own tool**,
using the existing L1 DSP estimators (`musicprobe/l1_baselines.py`):
1. **Pitch/F0 tracker** (monophonic) — phenomenon 1.
2. **Rhythm/beat tracker** (onset + tempo + meter) — phenomena 2 and 3 share
   this one; both are read off the same onset-grid measurement.
3. **Harmony/chord-key estimator** (chroma + Krumhansl correlation) —
   phenomenon 4.

Pitch and harmony are kept separate deliberately, even though both are about
pitch content: this project's own prior tracks already show they behave
completely differently (pitch's zoom+reference fix worked; harmony's 6
representations were a clean null or actively harmful) — collapsing them into
one tool would blur a distinction the project's own data already established.

**Known gap, harmony tool only**: `key_estimate`/`chord_quality_estimate` as
currently written compute one aggregate value for the *whole* clip — no time
resolution, so they can't localize *where* a modulation happens (needed for
worked example 2's per-bar tool report). Pitch and rhythm don't have this
problem — per-note pitch segmentation and onset detection are inherently
time-resolved already. Fix is a real but bounded piece of engineering: run the
same chroma-then-correlate logic per-bar instead of aggregating across the
whole clip. Not built yet.

## Ground-truth sourcing (per the "no musician review" constraint)

- **Synthetic/constructed items** (all 4 phenomena's counterfactual pairs):
  ground truth by construction — deterministic, no musician needed, same
  discipline as the whole existing battery.
- **Any real-recording anchor items**: ground truth by citation only
  (Suite A's already-proven citation-gate method), never an invented
  perceptual judgment call.

## What counts as success (sharpened 2026-08-21)

Not just "accuracy improved." Per `DIRECTION_DECISION.md`'s decisive-experiment
bar, a real result needs **all four**, together, not any one alone:
1. improvement on held-out real audio,
2. degradation when audio is swapped wrong,
3. degradation when the tool report is swapped wrong,
4. correct evidence spans, not just a correct final label.

The swap-sensitivity requirements (2, 3) are what separate "genuinely grounded
understanding" from "got better at guessing" or "learned a shortcut" — an
accuracy bump alone can't tell those apart. If any of the four is missing,
the honest thing to report is a narrower result, not "music understanding
improved." This applies once there's an actual intervention to test (a
training curriculum, or simply tool-report-in-context vs. not) — a step after
this pilot/schema exist, not part of building them.

## Decided (2026-08-21)

1. Worked example (modulation phenomenon) confirmed good as a starting point.
2. Tools: 3, one per facet (pitch/rhythm/harmony), using the existing L1 DSP
   estimators — see "Tools" section above.
3. Question phrasing: natural/product-like language — see "Phrasing
   convention" section above.

## Still open

1. Scoring: who/what checks whether a model's evidence field is "real" vs.
   confabulated — a fixed rule (does the cited tool-output string literally
   appear in the actual tool report?), or does this need a human read for the
   pilot's small N? Leaning toward a rule-based check for the easy cases +
   human read only for the ambiguous remainder, but not finalized.
2. The harmony tool's missing time-resolution (see above) needs to be built
   before worked example 2 (or any modulation-localization item) can actually
   run — not blocking the rest of the plan, but worth sequencing after
   whichever phenomenon gets built first if it isn't harmony.
