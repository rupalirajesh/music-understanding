# Runbook: Tracks L–Q (harmony) and R–W (rhythm) — for whoever runs this on the H100 box

Written 2026-08-05 for Sethu (or Sethu's Claude instance) to execute. This is the
companion brief to `scripts/17_run_tracks_lq.sh` and `scripts/18_run_tracks_rw.sh` —
those scripts are the literal command sequence; this document is the *why*, so you
can make good judgment calls instead of running commands blind. If you're an LLM
executing this: read this whole file before running anything, and use your judgment
at the checkpoints flagged below rather than mechanically running every line.

## 1. What this is and why it exists

Read `PROJECT_STATE.md` next actions 13 and 14 first for the full status table and
`RESEARCH_PLAN.md` §12.7 for the research narrative — this doc won't repeat all of
that, just the parts that matter for actually running it correctly.

Short version: Tracks C–H already established that when a model can't do a task from
audio alone, sometimes an explicit front-end representation fixes it (Track D-zoom: a
zoomed pitch-contour chart with a reference line took `tuning_judgment` from 0.53 to
0.89) and sometimes it doesn't (Track G: a flat chromagram was a clean null on the
whole harmonic cluster). Tracks L–Q and R–W are a **systematic sweep**, not a
single-shot test: six representations each, for two task clusters that have never
had a working fix (harmony: `key_id`/`mode_id`/`chord_quality`/`interval_id`) or any
causal intervention at all (rhythm: `tempo_bpm`/`beats_per_bar`, first LoRA arm of any
kind on this cluster).

**The sequence, and the idea behind each step** (harmony; rhythm is the direct
analogue — tempogram≈chromagram, onset-line≈pitch-contour, etc., see the table below):

| # | Track | What it is | What it's testing |
|---|---|---|---|
| L | peak-picked chroma | binarized chroma_cqt, top-3 pitch-classes/frame | does removing overtone blur (vs Track G's raw chroma) help? |
| M | zoomed peak-picked chroma | same, finer time resolution | does temporal resolution matter, independent of binarization? |
| N | harmony line | audio-derived multi-pitch scatter/trajectory | the direct generalization of the F0-contour that fixed pitch — does "show pitches as points on an axis" transfer from monophonic to polyphonic? |
| O | zoomed harmony line | same, y-axis zoomed to the active range | mirrors D-zoom's exact recipe |
| P | piano-roll | absolute pitch × time, note blocks | targets the *simultaneity* bottleneck (TASKS.md 1.3/2.3's arpeggiated-succeeds/block-fails pattern) — chords become visually separated rows instead of summed chroma energy |
| Q | tonal centroid (Tonnetz) | 6-D harmonic-distance projection of chroma | a geometrically different space (Harte/Sandler/Gasser 2006) where fifths/thirds are literally nearby — tests whether a music-theory-native geometry reads better than a raw pitch-class heatmap |

Rhythm (R–W) mirrors this exactly one step later: tempogram, peak-picked tempogram,
onset-strength line, zoomed onset-strength line, rhythm-roll (piano-roll equivalent,
targets the same "onsets need visual separation" idea but for beat timing), rhythm
necklace (Toussaint's circular rhythm-geometry representation, the Tonnetz analogue).

## 2. The policy that differs from every earlier track: **run all six, don't stop early**

Tracks C–H mostly followed "try something, if it works ship it, if not try the next
thing." **This batch is explicitly a comparison sweep** (Rupali's call, 2026-08-05):
the question isn't just "does anything fix this," it's "which representation creates
the richest usable signal, and does that answer generalize across the two clusters."
Run all 6 tracks per cluster to completion even if an early one (say, Track L or R)
already shows a strong positive result. The cross-track comparison step at the end of
each runbook script (`analyze_track_repr.py --track L --compare M N O P Q`) is the
actual deliverable, not just each track's individual number.

## 3. Before you start: what's already done vs. what you're doing

**Already done, verified, and committed** (CPU-side, no GPU needed, done on the
laptop 2026-08-05): all 12 renderers built and run against the full 1248-stimulus
battery (not just each cluster's own tasks — the `wrong_image` mechanism control
draws from the whole battery, so every representation needs full coverage or
`build_image_jobs` throws a missing-file error), all 12 job manifests built and
verified (correct held-out splits, zero train/held overlap, every task represented
in both). Three real bugs were caught during that local testing and fixed before
this ever reached the GPU box — see §5 below, worth knowing about since they explain
some of the design choices you'll see in the code.

**What you're doing**: the GPU training + eval steps (steps 3–4 in each runbook
script) — completely unverified on real hardware, since there's no GPU on the
laptop. Smoke-test every track before spending real compute on it, same discipline
as every earlier `gpu/` script in this repo.

## 4. Judgment calls while running

- **Smoke test looks wrong (loss is NaN, doesn't move, or the assertion in
  `assert_lora_applied` fires)**: stop, don't burn a real run on it. Compare against
  Track G's smoke-test behavior if you have that log — the LoRA config and training
  loop are byte-identical to Track G's (see `gpu/image_track_common.py`'s docstring,
  it's a diffed-and-confirmed extraction, not a rewrite), so if Track G's smoke test
  worked on this box, these should behave the same way. If it doesn't, the bug is
  more likely in environment/hardware than in this new code.
- **GPU OOM**: same batch size (1) and grad accumulation (8) as every other track in
  this project — if this box handled Track G/H fine, it should handle these. If not,
  check whether it's specifically the higher-resolution tracks (M, O — smaller hop
  length means more image pixels) before assuming it's a general problem.
- **Reading each track's result**: same statistics as every prior track —
  `image` vs `no_image` paired McNemar is the PRIMARY question (does it help), and
  `wrong_image`/`image_wrong_audio` are the MECHANISM checks (does the model actually
  read it, or is it ignoring the image / ignoring the audio). A track where accuracy
  goes up but `wrong_image ≈ no_image` (content doesn't matter) is a weaker result
  than one where `wrong_image` measurably hurts — that distinction is worth
  preserving in whatever you report back, not just the headline Δacc.
- **What counts as "worth flagging back to Rupali immediately" vs. "note and keep
  going"**: a large, significant (p<.05, CI excluding 0) positive Δacc on ANY
  harmony/rhythm task would be the first working fix either cluster has ever had —
  that's worth a heads-up before finishing the rest of the sweep, not just quietly
  logged. Nulls (expected — Track G was null, this is a real open question) don't
  need an interrupt, just document per the runbook's commit step.

## 5. Known limitations — read before over-interpreting results

- **Tracks V/W's `wrong_image` control is diluted.** ~28% of a random sample of the
  whole battery (56/200, verified) produces a near-blank rhythm-roll/necklace image
  (stimuli with fewer than 2 detected onsets — mostly sustained single tones from
  unrelated tasks, not click-based). This does NOT affect the primary
  `image`-vs-`no_image` analysis (0/160 of the rhythm cluster's own stimuli are
  sparse — they're all click tracks) — it only weakens the `wrong_image` mechanism
  check's statistical power for these two tracks specifically. If V/W's `wrong_image`
  delta looks weaker/noisier than other tracks', this is a plausible reason, not
  necessarily "the model ignores the necklace."
- **The rhythm necklace's cycle-length detection is a real but imperfect heuristic**:
  verified 12/15 (80%) correct on real stimuli spanning all 5 beats-per-bar
  categories (3/4/5/6/7), using median inter-onset-interval + an onset-strength-
  weighted circular-concentration statistic (see `scripts/render_rhythm_repr.py`,
  `_detect_click_period` and `render_rhythm_necklace`'s docstrings for the full
  story — two earlier, worse approaches were tried and rejected: raw envelope
  autocorrelation regularly locked onto a 2x/3x sub-harmonic of the true click rate,
  and unweighted onset-timing concentration can't distinguish n=3 from n=6 on a
  perfectly regular click train). This is a genuinely hard MIR problem (same reason
  `PROJECT_STATE.md` flags proper meter detection as needing essentia, not laptop
  heuristics) — 20% of Track W's necklaces are showing the wrong cycle length. Not
  fixed further; a plausible source of noise in Track W's numbers specifically.
- **Peak-picking (Tracks L/M/S) keeps exactly the top 3 bins per frame**, gated by a
  frame-energy floor (silent/near-silent frames — under 2% of that stimulus's peak
  frame energy — are left fully off, fixed 2026-08-05 after an independent review
  caught the original version fabricating content on silence). `TOP_K=3` was chosen
  to cover up to a triad without keeping too much overtone noise — if chord_quality's
  results look suspiciously flat across L/M, this fixed choice is a place to look
  before concluding the representation itself doesn't work.

## 6. Leakage discipline — the one rule that must never be violated if you touch this code

Every renderer takes only `(wav_path, out_path[, hop])` and derives everything from
the audio signal (or fixed constants) — never from `ground_truth`, `factors`, or
`task`. This was independently audited 2026-08-05 (grep-verified: no render function
receives or reads any manifest column beyond the audio path) and is clean. The one
place this is easy to get wrong if you extend the code: **Track V/W's grid/circle
size must come from a DETECTED periodicity** (`_detect_click_period`, audio-derived),
**never from the ground-truth `beats_per_bar` value** — that would hand the model the
answer directly (same trap as rendering sheet music, see `RESEARCH_PLAN.md` §12.2).
If you add a 7th representation to either sequence, re-derive everything from audio
the same way; don't take a shortcut through the manifest's answer columns even for a
"just testing" version.

## 7. Reporting back

Each track's `analyze_track_repr.py` run writes `results/trackA/track{x}_*_summary.csv`
+ `_graph.png`, same format as every prior track — these are what should get folded
into `PAPER.md`/`PROJECT_STATE.md` (ask Rupali's Claude instance to do the writeup,
or draft it yourself if you have the context — the existing Track G/H/D-zoom sections
in `PAPER.md` are the reference for the level of detail expected: headline Δacc + CI +
p, mechanism-control reading, and one sentence on what it implies for the
"universal representation" research question, not just a bare number). The final
cross-track comparison (`--compare`) is the thing to lead with — which representation
won, on which task, and whether the ranking is consistent between harmony and rhythm
or cluster-specific.
