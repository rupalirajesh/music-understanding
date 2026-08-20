# Direction decision — evidence-grounded music understanding

**Status:** proposed replacement for the "universal DSP representation" headline.

## The decision

We are **not** trying to make a model invent human music criticism from raw audio,
and we are **not** trying to prove that adding a spectrogram image makes models
"feel music." Those claims are vague, weakly testable, and the current experiments
do not support them.

We are building and evaluating an **evidence-grounded music analyst**. It may use
three legitimate sources of information, but must never blur them:

| Source | Appropriate answer | Required evidence |
|---|---|---|
| Retrieved knowledge | "Published analyses describe this modulation as …" | cited source |
| Audio measurement | "In this recording the tonic moves from … at 01:12" | timestamp + measurement/tool trace |
| Theory inference | "Given the measured chords, this functions as …" | stated rule + confidence |

This is exactly what a useful learning assistant should do. A student asking about
*Hey Jude* benefits from a sourced key, metre, tempo, and existing expert analysis.
There is no virtue in refusing text knowledge. The non-negotiable requirement is
that the assistant labels the answer's source and does not pass retrieved knowledge
off as listening, or a fragile audio guess off as established analysis.

## Paper claim (one sentence)

**Current music LALMs can sound knowledgeable without being audio-grounded; we
measure this separately, localize the failure, and test whether auditable music
tools—not an undifferentiated extra modality—produce reliable, transferable musical
analysis.**

## What is and is not novel

### Not a contribution

- Giving a model a spectrogram, chromagram, F0 contour, MIDI, or another modality.
  These representations and audio+vision/audio+symbolic systems already exist.
- Showing a model can state familiar facts about famous songs. That is retrieval or
  pretraining knowledge, and is product value rather than listening evidence.
- A classifier-style result such as "which first note is higher?" It is a necessary
  perceptual sanity check, not evidence of broad music understanding.
- Claiming that basic synthetic training improves real-music analysis before a
  controlled held-out transfer experiment demonstrates it.

### Potentially publishable contribution

1. **Source-separated evaluation.** Each answer is attributed to retrieval, audio,
   and theory inference; counterfactual controls show which source actually changed
   the answer. Existing music QA rarely tests all three together.
2. **A causal, auditable assistant protocol.** Require a timestamped observation /
   retrieved citation / explicit theory rule with every analytical claim. Score both
   answer correctness and evidence correctness.
3. **A mechanism result, not just a leaderboard.** With L1 DSP, L2 probes, L3
   behaviour, no-audio, wrong-audio, and tool ablations, say whether a failure is
   missing signal, failed audio-to-language readout, or failed reasoning.
4. **Transfer only if earned.** Test whether a curriculum or tool interface trained
   on controlled audio improves *unheard, legally usable, real recordings* while
   remaining sensitive to the audio and to the tool content.

The current F0-zoom result belongs in (3) as a sharp boundary condition: a readable,
referenced visual interface can rescue an absolute-pitch judgement. The null results
for spectrograms, chromagrams, and rhythm representations show that this is **not**
evidence for a universal visual-DSP solution. Do not make it the headline.

## The product and the research are different

| Layer | User-facing behaviour | What the paper evaluates |
|---|---|---|
| Knowledge | Finds and cites expert material about a known song | factual accuracy + source fidelity; no audio credit |
| Listening | Measures the uploaded recording | counterfactual audio grounding + calibrated uncertainty |
| Analysis | Explains musical consequence from the measured facts | correctness of the inference and its evidence chain |
| Teaching | Adapts the explanation to a learner | optional product study, not the core technical claim |

The final product should combine all four. The paper's central result is whether it
knows which layer it used and can ground the listening layer reliably.

### Provenance is system-enforced, not self-reported

Do **not** ask a bare LALM “where did you get that answer?” and trust its reply.
Its internal text, audio, and training knowledge are entangled; a fluent
post-hoc explanation is not evidence. Instead, make the assistant operate over
named, logged inputs:

1. A retriever returns source passages with stable IDs and quotations.
2. Audio tools return structured measurements with clip IDs, time spans, values,
   and uncertainty (for example `beat_17=00:42.6`, `chord_candidate=V7, 0.63`).
3. The LLM writes an answer that can cite only those IDs and explicitly applies a
   theory rule to tool-derived observations.
4. Evaluation swaps the audio, the tool report, or the retrieved passage. The
   claimed provenance is accepted only if the answer changes in the predicted way.

The assistant should normally combine sources. Retrieval can add historical context,
audio establishes what happens in *this recording*, and theory explains the musical
relationship. When they disagree, it must surface the disagreement rather than vote:
a web source may describe the original recording while the upload is a cover,
transposed performance, edit, or differently tuned release.

## Minimal benchmark: three suites, not "all music"

Do not attempt exhaustive coverage of genres, every music-theory question, and every
commercial recording. A benchmark is a set of **precise claims** whose evidence can
be checked, with deliberately selected stress tests.

### Suite A — knowledge only (retrieval)

- Famous-song facts and published analyses: key, tempo range, instrumentation,
  credited analysis, historical/contextual claims.
- Input: text/title plus an approved source collection; **no audio required**.
- Score: answer accuracy, citation entailment, and correct source label.
- Purpose: establish that text priors are useful and valid, while preventing them
  from inflating a listening score.

### Suite B — audio only (counterfactual perception)

- Original/commissioned or synthetic-but-musical excerpts with hidden titles.
- Same excerpt is altered one musical variable at a time: transposition, local key
  change, chord quality/function, interval contour, beat grouping, onset location,
  tempo change, tuning deviation, instrument entrance.
- Questions ask for relational musical claims, not merely a plot-reading fact:
  *where does the modulation occur; which version preserves the motif under
  transposition; which chord change changes function; is the syncopation moved?*
- Score: accuracy, no-audio and wrong-audio drops, paired counterfactual consistency,
  timestamp error, and confidence calibration.

### Suite C — analysis from audio + tools

- The model receives audio and a deterministic tool report (for example: beat grid,
  pitch track, chord candidates with uncertainty), not an answer-revealing score.
- It must return: `claim | evidence span/tool output | theory rule | confidence`.
- Ablate audio, tool, and replace the tool report with one from another clip.
- Score source attribution separately from final correctness. A correct answer with
  a false timestamp or invented measurement is a failure.

Start with 6–8 phenomena and 30–50 items each. Expand only after piloting task
reliability with musicians. This is sufficient for a focused paper; it is not a
claim to represent all musical cultures.

### First pilot: six pre-registered phenomena (Western tonal scope)

| Phenomenon | User-relevant claim | Counterfactual pair | Required evidence |
|---|---|---|---|
| Local modulation | "Does it move key; from/to what; where?" | Same excerpt, one pivot/target key changed | onset/measure span + tonic/chord evidence |
| Harmonic function | "Why does this chord feel like it leads home?" | Same chord quality, changed surrounding progression | local chord window + stated function rule |
| Motif invariance | "Is this later phrase the same idea, transposed?" | Preserve contour/intervals vs alter one interval | both spans + interval relation |
| Syncopation | "Where is the rhythmic tension and what creates it?" | Shift onset off/on the beat without changing tempo | beat grid + onset locations |
| Tempo/metric change | "Does the pulse or grouping change here?" | Preserve audio texture while changing BPM or 3/4-vs-6/8 grouping | beat/downbeat spans + uncertainty |
| Intonation/expressive pitch | "Is that note bent/out of tune relative to the local reference?" | Same phrase with a controlled detune/bend | pitch trace + stated reference |

These do **not** need to be solved by one generic representation. Each needs a
task-appropriate deterministic measurement. The scientific question is whether the
language model can use those measurements honestly and compose them into a correct
musical explanation.

## The decisive experiment: does basic training transfer?

This is a hypothesis, not the story yet.

1. Train on Suite-B-style controlled examples for a small set of primitives.
2. Hold out instruments, keys, tempi, soundfonts/rooms, composers, and all exact
   progressions. Use recordings made after the training materials are frozen.
3. Evaluate on original real performances in Suite C, where the requested claim
   combines primitives (for example, modulation **and** location **and** function).
4. Compare four matched arms:
   - base LALM,
   - LALM fine-tuned on question/answer examples,
   - LALM plus tool report,
   - fine-tuned LALM plus tool report.
5. Require all of: improvement on held-out real audio; degradation with wrong audio;
   degradation with a wrong tool report; and correct evidence spans. If any is
   absent, report a narrower result, not "music understanding improved."

The best likely outcome is not "basic concepts make it feel music." It is:
**a small, controlled perception curriculum and/or an auditable tool interface
improves a defined family of audio-grounded analytical claims, with evidence that
the improvement genuinely transfers.**

## Relationship to existing work

- **MuChoMusic / RUListening:** established that music QA can be solved from text
  priors; RUListening hardens multiple-choice distractors. We retain the key lesson
  but add per-claim source attribution and audio/tool counterfactuals.
- **CMI-Bench:** broad real-world MIR instruction following with conventional MIR
  metrics. It measures useful tasks, but does not answer whether an explanation was
  grounded in the relevant evidence route.
- **Core Music Perception Tasks:** closest conceptual neighbour. It compares audio
  with MIDI on syncopation, transposition, and chord quality, and finds strong
  symbolic performance but brittle listening. Our paper must explicitly extend—not
  repeat—that result: causal source controls, tool interfaces, evidence traces, and
  held-out transfer.
- **MUSE / MusICA-MetaBench:** important relational-perception benchmarks, with
  human baselines or modality comparison. Use them as external validation, not a
  substitute for the provenance/assistant test.
- **MusTBench:** recent work makes temporal grounding its central problem. Avoid
  presenting timestamps alone as novel; our distinction must be *musical claim +
  provenance + causal evidence*, with timestamps as one required component.
- **Music Flamingo:** already claims broad music-specialized understanding and
  strong benchmark scores. A new paper cannot merely say a model gets high scores;
  it must demonstrate which scores are listening, retrieval, or reasoning and why.

## Data and licensing decision

- **Training + released benchmark audio:** use self-recorded/commissioned audio,
  public-domain material with a new recording you own, or files whose licence
  explicitly permits the intended redistribution and ML/research use. Record the
  licence and provenance per item.
- **Commercial songs:** may be useful as private product demonstrations or a
  separately permissioned evaluation, but do not put clips in a released training
  set or public benchmark without rights. A short excerpt is not automatically safe;
  fair-use/fair-dealing analysis is jurisdiction- and fact-specific.
- **Culture/genre claim:** make no "all genres" claim from a Western tonal pilot.
  State scope honestly and add a separately designed, expert-led non-Western module
  only when its concepts and annotation standards are appropriate.

## Immediate actions

1. Freeze the paper title/RQ around **evidence-grounded music analysis**, not a
   universal representation or visual multimodality claim.
2. Preserve the existing L1/L2/L3 and control results as the diagnostic backbone.
3. Stop new generic representation-ladder experiments unless they test Suite C's
   causal protocol and a pre-registered musical phenomenon.
4. Define the first six phenomena and obtain musician review of the labels and
   explanations before generating more stimuli.
5. Implement the evidence schema and ablations before doing any training run.
6. Run MUSE/Core/MusICA as external checks only after the internal protocol works.
