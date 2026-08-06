# Runbook: Tracks X (harmony) and Y (rhythm) — for whoever runs this on the H100 box

Written 2026-08-06, the direct follow-on to `RUNBOOK_tracks_lq_rw.md`. Read that
document first if you haven't — this one assumes the same context (Tracks L-Q/R-W's
results, the registry-based harness in `gpu/image_track_common.py`, the leakage
discipline) and only covers what's new. Companion to `scripts/19_run_tracks_xy.sh`,
same relationship as before: that script is the command sequence, this is the why.

## 1. Why these two tracks exist

Tracks L-Q and R-W (full 3-seed results now landed, see PAPER.md/PROJECT_STATE.md)
were a clean null across twelve representations — no harmony or rhythm chart beat
audio-only, and several (key_id, tempo_bpm) were actively *hurt* by adding an image.
That looked, at first pass, like "the D-zoom trick doesn't generalize past pitch."

Look closer at what D-zoom actually was, though: it wasn't just "zoom in" (Track D
force tried a sharper/forced-attention image alone — null) and it wasn't just "add a
reference" (Track H tried an in-audio reference tone alone — null). It was **zoom
and an explicit annotated reference position, together** — neither ingredient worked
by itself for pitch either, across three earlier attempts. The L-Q/R-W ladder tested
zoom (M, O, U) and reference/structural richness (P, V) as *separate* items on a list
of six independent ideas, the same way D force and Track H each tested one ingredient
alone. Nobody has yet tried the combination on harmony or rhythm. That's what X and Y
are — not a 13th and 14th independent guess, but the specific missing cell in the
2x2 (zoom × reference) that actually mattered for pitch.

| | no explicit reference | explicit reference |
|---|---|---|
| **not zoomed** | Track L / R (baseline ladder start) | Track P (piano-roll) / Track V (rhythm-roll) |
| **zoomed** | Track M,O / Track U | **Track X / Track Y ← untested until now** |

- **Track X — zoomed peak-picked chroma + estimated-tonic reference row.** Base:
  Track M's chroma at the finer `HOP_ZOOM` time resolution. Added: the tonic pitch
  class is estimated from THIS stimulus's own chroma via Krumhansl-profile
  correlation (`musicprobe.l1_baselines`'s exact method, restricted to the tonic —
  mode is a nuisance parameter here, discarded), and highlighted as a shaded
  band + labeled line on the chart — the harmonic analogue of D-zoom's red
  "in tune" line. `scripts/render_harmony_repr.py:render_chroma_zoom_ref`.
- **Track Y — zoomed rhythm-roll.** Base: Track V's onset-vs-detected-pulse-grid
  chart, rendered at Track U's finer `HOP_ZOOM` resolution instead of the default
  hop (this one only needed a resolution bump — Track V already carried an
  audio-derived reference grid, it just had never been combined with zoom).
  `scripts/render_rhythm_repr.py:render_rhythm_roll` (now takes a `hop` arg,
  registered as `rhythm_roll_zoom`).

## 2. Non-leakage — verified, same standard as every prior track

Both estimates are audio-derived, never from `ground_truth`/`factors`/`task`:
- Track X's tonic estimate: Krumhansl correlation against THIS stimulus's own
  `chroma_cqt`, same method (and same profiles, imported from
  `musicprobe.l1_baselines`) as the L1 DSP floor's `key_estimate` — not the answer
  column, an independent re-derivation from the audio, same discipline as
  D-zoom's pyin-estimated (not ground-truth) reference pitch.
- Track Y's pulse grid: reuses `_detect_click_period` unchanged (median inter-onset
  interval from librosa's own onset detector) — the exact same function Track V
  already used and that was independently leakage-audited 2026-08-05.

Verified locally 2026-08-06 (CPU, no GPU needed):
- Both renderers run 1248/1248 stimuli, 0 errors, whole-battery scope (required for
  a valid `wrong_image` draw pool, same reason as every earlier track).
- Held-out splits built and checked directly: **Track X = 287 train / 612 held**,
  identical to Track G/L-Q's split (expected — same harmony task set, same
  `_held_out_mask` soundfont/base_midi logic, no change to the split rule). **Track Y
  = 127 train / 132 held**, identical to Track R-W's split. Zero train/held overlap
  in either. Every target task present on both sides of both splits.

If you extend either renderer further, the one rule that must never be violated
(RUNBOOK_tracks_lq_rw.md §6 has the full version): derive any annotation from the
audio signal only, never from the manifest's answer columns.

## 3. What's already done vs. what you're doing

**Already done, verified, committed** (laptop, CPU-only, 2026-08-06): both renderers
written, registered in `gpu/train_track_repr.py`'s `TRACKS` dict as `"X"`/`"Y"`, run
against the full battery, held-out splits verified as above.

**What you're doing**: steps 3-4 of `scripts/19_run_tracks_xy.sh` — GPU training +
eval, completely unverified on real hardware (no GPU on the laptop). Smoke-test each
track before spending real compute, same as always.

## 4. Reading the results

Same statistics, same script, as every earlier track (`gpu/analyze_track_repr.py`
already handles X/Y with zero changes — it's registry-driven off `TRACKS`). Two
questions matter here specifically, more than for a fresh independent guess:

1. **Does X/Y beat every L-Q/R-W variant that was tried alone?** The run script's
   last two steps (`--compare` against the full ladder) are the actual deliverable.
   If X or Y shows a significant positive Δacc where none of L-Q/R-W did, that
   confirms the "combination, not the individual ingredients" reading of D-zoom, and
   is worth flagging back immediately — same "worth an interrupt" bar as
   `RUNBOOK_tracks_lq_rw.md` §4 (large significant positive = first working fix for
   that cluster).
2. **If X/Y is ALSO null**, that's an equally real result: it means the D-zoom
   mechanism (zoom + reference) is pitch-specific for a deeper reason than "nobody
   combined the ingredients yet" — e.g. pitch is a single continuous scalar that
   maps onto one chart position, while key/chord/tempo/meter are categorical or
   aggregate judgments that a single reference marker can't resolve the same way.
   Document it with the same rigor as an L-Q/R-W null (mechanism controls, not just
   the headline number) — this closes off the "did we even try the right thing"
   question either way.

## 5. Reporting back

Same format as `RUNBOOK_tracks_lq_rw.md` §7 — `results/trackA/trackx_*_summary.csv`
/ `tracky_*_summary.csv` + graphs, folded into PAPER.md/PROJECT_STATE.md with
headline Δacc + CI + p, mechanism-control reading, and how it updates the "does the
D-zoom trick generalize" question from RESEARCH_PLAN.md §12.7.
