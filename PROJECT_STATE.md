# PROJECT STATE — the everything doc (pick-up-where-we-left-off)

Last updated: 2026-08-05. Update this whenever anything changes hands.

## Machines & workflow
- **Laptop** (no GPU): stimulus synthesis, API evals, scoring/analysis,
  stimulus auditing by ear, manual response verification.
- **H100 box**: open-model runs, activation extraction/probing, generation
  battery; outputs are committed back to `results/` and pushed.
- Repo: https://github.com/rupalirajesh/music-understanding (PUBLIC — keep
  docs free of personal details).

## What exists (all pushed, commit a6ae042 + later)
- `RESEARCH_PLAN.md` — full background primer + study design (L1/L2/L3,
  tracks A/B/C, hypotheses H1–H6). Read Part 0 first.
- `experiments/TASKS.md` — the tier list: every task, N, controls, what a
  pass/fail means, sample-size rationale.
- `experiments/musicprobe/` — package: generators, manifest, prompts (3
  paraphrases + diagnostic distractors), jobs (controls), runners (API +
  H100 local), scoring (accuracy/audio_gain/confusions/psychometrics), DSP
  L1 baselines.
- `experiments/manifests/` — FROZEN battery v1: 1,176 stimuli / 2,063 jobs.
  Do not regenerate while v1 runs are in flight; changes = v2.
- `experiments/gpu/` — Track B: activation extraction (MERT/Whisper/CLAP
  loaders done; AF3/Qwen-Omni TODO) + held-out-soundfont linear probes.
- `experiments/genmodel/` — MusicGen constraint battery + DSP adherence
  scoring; single-constraint rule; vocab-vs-theory pairs. Not yet run.
- `experiments/scripts/` — 00 soundfonts, 01 generate (seeded, machine-
  stable), 02 jobs, 03 listening page, 04 review export, 05 self-test
  (14 checks, ALL PASSING as of 2026-07-18).
- Docs workflow: a local read-this inbox file (not in git, cleared
  iteratively), `PAPER.md` (results skeleton), this file (state).

## Status of runs (updated 2026-07-22)
| What | Where | Status |
|---|---|---|
| Qwen2-Audio-7B full battery | H100 box | done |
| Qwen2.5-Omni-7B, Qwen3-Omni-30B, AF3, Music-Flamingo full battery | H100 box | done (landed commit 4aa5dcf) |
| Gemini-2.5-Pro battery (via Portkey) | H100/laptop → API | done |
| GPT-4o-audio | — | **out of scope — no OpenAI API access** (removed from the plan 2026-07-25) |
| MOSS-Music-8B-Instruct (new, added 2026-07-25, Sethu's initiative) | H100 box | full Track A battery done |
| MERT/Whisper/CLAP extraction + probes | H100 box | done — all 3 encoders × 11 tasks |
| Own-encoder re-probe (key_id/mode_id/chord_quality/interval_id) | H100 box | done 2026-07-24, commit 83c722c |
| Attention diagnostic (`gpu/attention_audio.py`) | H100 box | **DONE CORRECTLY 2026-07-25** (commit c348ea6, eager-attention verified, all 5 models, `--per-task 6`) — supersedes the 2026-07-24 retracted run. See Known gaps for the corrected finding. |
| Microtone probe (relative-pitch direction + absolute detune, new task, Sethu's initiative) | H100 box | done 2026-07-25, commit b0dec99 |
| Track C (3-arm LoRA on AF3: llm_only / llm_encoder / control) | H100 box | **done 2026-07-26** (commit `a37cc8a`) — `octave_id`/`note_count` alignment-fixable (+0.50/+0.59 and +0.43/+0.40 over baseline); `tuning_judgment`/`cents_discrimination` resist fine-tuning even with the encoder also tuned |
| Track D Phase 1 (Qwen2.5-Omni-7B + spectrogram image, single LoRA arm) | H100 box | done 2026-07-26 (commit `a37cc8a`) — first-pass "image helps cents" (0.67→0.77) turned out to be an OOD training confound; see conclusive rerun below |
| Track D conclusive (mixed-condition training, paired 3-seed McNemar) | H100 box | done 2026-07-28 (commit `d7d4c3f`) — precise null: spectrogram adds nothing on any task (Δacc 95%CI includes 0 everywhere); mechanism controls confirm the model ignores it |
| Track D force (same spectrogram + modality-dropout training) | H100 box | done 2026-07-28 (commit `841da35`) — forces the model to actually use the image (wrong-image now craters perf, p=.003/.016) but accuracy still doesn't improve; spectrogram lacks fine cents-level detail |
| Track D zoom (zoomed F0-contour image + in-tune reference line) | H100 box | done 2026-07-29 (commit `03c7bde`) — cents 0.55→0.94 (p<1e-4), tuning 0.53→0.89 (p<1e-4); first method that fixes absolute tuning |
| Track E (pitch-tracker output as text in the prompt, audio-only) | H100 box | done 2026-07-29 (commit `03c7bde`) — cents 0.62→0.92 (p<1e-4), scalable/deployable (no image needed); tuning ns (text has no reference point) |
| Track F (learned pitch-stream fusion: trainable projector injects F0 features into embedding space) | H100 box | done 2026-07-29 (commit `2418e80`) — null; injection verified to reach the model (logit shift scales with pitch value) but behaviorally ignored — likely a data-size limit (~348 training examples to learn a new modality interface from scratch) |
| MusicGen battery | H100 box | done — tempo/key/register scored; meter/mode deliberately left for manual scoring |
| Track G (chromagram front-end for key_id/mode_id/chord_quality/interval_id — first causal test on the harmonic cluster; Tracks C-F only ever targeted pitch/tuning) | H100 box | **done 2026-08-05** (commit `8050378`, Sethu) — null on all 4 tasks (3-seed paired McNemar): key_id Δ=−0.07 (p=.26), mode_id Δ=+0.04 (p=.57), chord_quality Δ=+0.13 (p=.11, n=72), interval_id Δ=+0.03 (p=.64); all CIs include 0. Mechanism controls: wrong-chromagram ≈ no-chromagram everywhere (content not read); wrong-audio+chromagram only craters key_id (p=.02). Note: this tested ONE variant — a force-style-trained but flat, unannotated, unzoomed chromagram — the analogue of Track D's early "conclusive/force" stage, not the full iteration that eventually found D-zoom. The zoomed/annotated step that actually rescued pitch has not been tried for harmony yet; see next action 13. |
| Track H (in-audio reference tone for tuning_judgment — tests whether Track D-zoom's "needs an explicit reference" finding works delivered in-AUDIO instead of switching modality) | H100 box | **done 2026-08-05** (commit `8050378`, Sethu) — flat null: reftone vs plain Δ=+0.01 (p=1.0); wrong_reftone vs plain Δ=+0.02 (p=.82) — a WRONG reference doesn't mislead any more than a correct one, so the model isn't comparing target-to-reference in audio at all. D-zoom's reference fix is visual-channel-specific, not a general "give it a reference" effect. |
| Track F-aug leakage fix (rerun of the 9x-data pitch-fusion aug run) | H100 box | **done 2026-08-05** (commits `8050378`+`b54c4bf`, Sethu) — `generate_aug.py` was sampling training pitches into the held-out eval band; capped at `MAX_TRAIN_MIDI=70` and reran. Corrected audio-only-on-held-out: cents 0.62→0.78→**0.74** (clean, vs 0.89 leaky) — about half the leaky jump was real. tuning 0.51→0.68→**0.62** (clean, vs 0.83 leaky) — nearly halved, now ≈ 2AFC majority rate, NOT evidence of learned absolute-tuning perception. Fusion-null verdict unchanged (model still ignores the fused stream). |
| AF3 / Music Flamingo loaders | code | done, verified working |
| Qwen2-Audio published-number replication | — | still TODO; pick exact benchmark subset first |
| Analysis pass on all of the above | laptop | first pass done 2026-07-22 — see PAPER.md Results; dashboard + plots in `experiments/results/trackB/analysis/`; attention + microtone graphs in `experiments/results/trackB/attention/attention_graph.png` and `.../probes/microtone_probe_graph.png` (2026-07-25) |

## Decisions made (and why) — chronological
1. Only fine-tuning is feasible (no pretraining budget); the study's job is
   to find WHERE fine-tuning helps (alignment/data gaps) vs can't (encoder).
2. Test generation models too — may end up fine-tuning one. Understanding
   measured as constraint adherence scored by our own DSP.
3. Synthetic stimuli for Tiers 1–2 (unlimited N, perfect ground truth,
   factor control); real datasets only for Tier 3.
4. Meter task = beats-per-bar (audio can't convey notated denominator —
   caught in stimulus audit). 3/4-vs-6/8 grouping task specced separately.
5. Key task "cadence" form renamed "progression" (it's not cadence-type ID);
   cadence_type specced as future task 2.6.
6. tuning_judgment added (12-TET grid probe, behavioral half).
7. All responses stored verbatim; response-level verification is done
   manually (review CSVs + explanations.csv). Never trust auto-parse alone.
8. Single-constraint generation prompts only; compound later.
9. WAVs/soundfonts not in git; regeneration is deterministic (stable crc32
   seeding — NEVER use builtin hash() for seeds).
10. Battery v1 frozen at 2,063 jobs. Ladder arm will be battery v2.
11. NEW DIRECTION (2026-07-18): representation ladder → universal
    music representation → world-model framing. 5 requirements: inferable
    from audio / token-compact / sufficient for battery / human-readable /
    genre-universal incl. continuous pitch (staff notation fails #5; that's
    the Carnatic module's future role as stress test).
12. Tracks C–F (2026-07-26 to 07-29): the alignment-gap hypothesis (H4) is only half
    right for pitch. LoRA on the existing audio→LM pathway (Track C) fixes readout-level
    gaps (`octave_id`, `note_count`) but not microtone perception, even with the encoder
    also tuned. Microtone perception turned out to be fixable, but only by routing it
    through a representation the model already knows how to read — pitch-tracker output
    as **text** (Track E: fixes relative pitch, no image needed, cheap/deployable) or a
    **zoomed chart with an explicit reference line** (Track D-zoom: the only method that
    also fixes absolute tuning). An end-to-end learned fusion into embedding space
    (Track F) was a clean null at this data scale (~348 examples) — bolting on a raw
    feature and hoping a small adapter learns to use it does not work; reusing an
    existing, pretrained interface (numbers, images) does. Treat this as the working
    default for any future modality-injection experiment, not just pitch.
13. Tracks G/H (2026-08-05, run and landed): two more front-end candidates in the same
    spirit as C-F, chosen to extend into a cluster and a delivery mechanism neither C-F
    touched. Track G asked whether the "give it a rendered chart" idea (Track D)
    generalizes past pitch to the harmonic cluster (`key_id`/`mode_id`/`chord_quality`/
    `interval_id`) via a chromagram (12 pitch-class rows x time — the harmonic analogue of
    the F0-contour). Track H asked whether Track D-zoom's "needs an explicit reference"
    finding for absolute tuning can be delivered in-AUDIO (mix a reference tone into the
    clip itself) rather than switching modality to vision. **Both came back null** — see
    Status table for numbers. Reading: the D-zoom reference-line fix does not generalize
    into "any chart helps" or "any reference helps" — it's specific to an annotated visual
    position for pitch. Harmony (key/mode/chord/interval) remains an open gap with no
    working front-end tried yet. Both runs reused the established discipline (dropout-style
    training so eval conditions stay in-distribution, wrong-condition mechanism controls,
    held-out splits, paired McNemar over 3 seeds) rather than repeating Track D Phase 1's
    single-arm mistake, so the null results are trustworthy, not a methodology artifact.

## Next actions (ordered, updated 2026-08-05)
GPT-4o-audio removed from this list entirely 2026-07-25 — no OpenAI API access, out of
scope. 6 models (Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni-30B, AF3, Music-Flamingo,
Gemini-2.5-Pro) + MOSS-Music-8B is the Track A roster unless that changes.

1. ~~Re-run the attention diagnostic on the 4 newer open models~~ **DONE 2026-07-25**
   (commit c348ea6, eager-verified). See Known gaps above for the corrected findings.
2. ~~Sanity-check `beats_per_bar` and `mode_id` by hand~~ **DONE 2026-07-31** — no scoring/
   parsing bug (both use the same generic open-format substring match and the same 10%
   `WRONG_AUDIO_FRACTION` design as every other task, verified against `jobs.py`). Re-derived
   the wrong-audio deltas directly from `scored__*.parquet`: beats_per_bar's "inversion" is
   real in direction (mean +0.156 across 7 models, 5/7 positive) but its control is only
   n=14/model (std 0.141) — about 1 SE, not the clean signal PAPER.md previously implied.
   mode_id is weaker still (+0.070, std 0.169) — within noise, no action needed. Revised
   PAPER.md language to "low-confidence, directionally suggestive," not "structurally
   broken." Before any future LoRA arm on beats_per_bar, enlarge the wrong-audio sample
   first (bump `WRONG_AUDIO_FRACTION` for this task or pool paraphrases) — n=14 isn't
   enough to certify or refute the task.
3. ~~Re-probe each LALM's OWN encoder~~ **DONE 2026-07-24** (commit 83c722c,
   `gpu/extract_activations.py --own-encoder`, submodule paths verified:
   `thinker.audio_tower` for Qwen-Omni, `model.audio_tower` for Flamingo). Result: own-encoder
   probe accuracy on `key_id`/`mode_id`/`chord_quality`/`interval_id` stays modest —
   comparable to, not clearly beating, the generic MERT/Whisper/CLAP baselines. Reading
   unchanged from before the re-probe: behavioral success on these 4 tasks is more likely
   priors than a richer internal representation. No further action.
4. ~~Run the L1 DSP floor~~ **PARTIALLY DONE 2026-08-05** (laptop, pure numpy/scipy, no
   essentia needed — `musicprobe/l1_baselines.py`) — extended from 4/13 to 10/13 tasks.
   New estimators: `octave_estimate` (reuses `f0_autocorr`), `tuning_estimate` (nearest-
   12-TET-grid distance, 25¢ threshold), `note_count_estimate`/`interval_estimate`
   (FFT peak-picking with greedy harmonic-series rejection), `chord_quality_estimate`/
   `mode_estimate` (extend `key_estimate`'s Krumhansl-correlation method to CHORDS'/MODES'
   binary templates x 12 roots/tonics). Actual run, `python -m musicprobe.l1_baselines`:
   | task | L1 acc | n | chance | note |
   |---|---|---|---|---|
   | octave_id | **1.00** | 72 | — | trivial from audio; matches Track C's LoRA-fixable verdict |
   | cents_discrimination | 1.00 | 180 | — | (existing) |
   | pitch_note_id | 1.00 | 72 | — | (existing) |
   | tempo_bpm | 0.82 | 60 | — | (existing) |
   | key_id | 0.80 | 96 | ~4% (24-way) | (existing) |
   | chord_quality | 0.60 | 96 | 12.5% (8-way) | strong recoverable signal |
   | tuning_judgment | 0.63 | 120 | 50% | above chance but noisy — naive threshold, not tuned |
   | interval_id | 0.54 | 144 | ~8% (12-way) | recoverable, well above chance |
   | note_count | 0.40 | 100 | ~20-30%ish | noisy heuristic, real headroom, not a hard ceiling |
   | mode_id | **0.25** | 104 | ~8% (13-way) | weakest of the six — barely 3x chance, and this now lines up with L2 (own-encoder probe also barely-above-chance for mode_id) AND L3 (behavioral ~chance) — three independent methods agree mode_id is the hardest task in the battery. Corroborates the existing known-gap note that mode melodies are random diatonic walks, not musician-composed — may be a stimulus-quality issue as much as a model one. |
   Still not covered: `beats_per_bar`, `progression_id` (need real beat/chord-sequence
   tracking, autocorrelation-only isn't trustworthy as a floor), `instrument_id` (already
   near-ceiling behaviorally, L1 floor isn't the interesting question there) — still
   genuinely blocked on essentia/madmom on the H100/Linux box (confirmed 2026-07-31:
   `pip install essentia` has no wheel for the laptop's Python/platform).
5. ~~Qwen2-Audio published-number replication~~ **RESOLVED 2026-08-05, still needs the GPU
   run**: confirmed via primary source (fetched the actual MuChoMusic paper, arxiv
   2408.01337, Table 3 — not a secondhand summary) that it evaluates **Qwen-Audio v1**
   (arxiv 2311.07919) at 51.4% overall / 51.1% knowledge / 51.0% reasoning / 89.7% IFR.
   **It does not evaluate Qwen2-Audio at all** — checked a 2025 follow-up (arxiv 2504.00369)
   that references Qwen2-Audio too; it only cites v1's number secondhand, no standalone v2
   result anywhere. So there is no published Qwen2-Audio MuChoMusic number to replicate —
   the original next-action framing was chasing a number that doesn't exist. Correct fix:
   point the harness at **Qwen-Audio v1** (`Qwen/Qwen-Audio-Chat` on HF) instead, since
   that's the only model with a citable ground truth for this benchmark; this is a one-off
   harness sanity check, not a Track A roster change (Qwen2-Audio stays as-is in Track A).
   Scaffold written: `gpu/eval_muchomusic.py` (dataset `mulab-mir/muchomusic` via HF
   `datasets`). **Unverified** — `datasets`/`huggingface_hub` aren't in the laptop venv, so
   this hasn't executed end to end; field names are a best guess from the dataset card, not
   confirmed against the real schema.
   **Deprioritized 2026-08-05 (Rupali's call)**: not worth running as originally scoped.
   The only way this buys real harness-validation value is against Qwen-Audio v1
   specifically (the one model with a citable number) — running it against Qwen2-Audio,
   Qwen3-Omni, or any other roster model would produce a number with nothing to check it
   against, i.e. not actually a validation. Script is left in place in case a v1 run is
   ever wanted as a cheap one-off sanity check; not queued as active work.
6. Ladder arm (battery v2) — L1 features into prompts at one-abstraction-below-answer,
   features-only + few-shot variants; keep v1 job_ids untouched. Deferred behind 1–5.
   Note (2026-07-29): Tracks D-zoom/E are effectively a one-task preview of this arm
   (in-context F0 feature, text and image forms) — worth reusing their harness rather
   than rebuilding from scratch when this starts.
7. ~~Track C is set up, not yet run~~ **DONE 2026-07-26** (commit `a37cc8a`) — see Status
   table. `octave_id`/`note_count` alignment-fixable; `tuning_judgment`/`cents_discrimination`
   resist LoRA even with the encoder tuned — carried forward into Track D/E/F below.
8. ~~Track D Phase 1 is set up, not yet run~~ **DONE, then superseded 3x, 2026-07-26 to
   07-29** — Phase 1's apparent win was an OOD confound (commit `d7d4c3f` conclusive
   rerun: true null). Forcing image use didn't help either (commit `841da35`: mechanism
   works, accuracy doesn't). A zoomed F0-contour + reference-line image finally fixed
   both cents and absolute tuning (commit `03c7bde`, Track D-zoom). Parallel text-only
   front-end (Track E, same commit) fixes cents without an image but not tuning. An
   end-to-end learned fusion alternative (Track F, commit `2418e80`) was a null — the
   model ignores a raw feature injected into embedding space, likely too little data
   (~348 examples) to learn a new modality interface from scratch. See Status table and
   PAPER.md Results for full numbers.
9. ~~Wire up before/after L2 probes on the Track D-zoom / E checkpoints~~ **RESOLVED
    ANALYTICALLY 2026-07-31, no GPU run needed** — `build_lora_config()` (shared by
    train_track_d/_force/_conclusive, train_track_e_f0text, train_track_f_pitchfuse) only
    matches `target_modules` under `thinker.<lm_path>` (the LLM decoder); the regex never
    reaches `audio_tower` or the vision tower, so those receive zero gradient in every
    Track D/E/F run — the encoder's own representation is provably unchanged (bit-identical
    forward pass pre/post fine-tune). Re-running `extract_activations.py --own-encoder`
    against these checkpoints would just reproduce Track B's existing numbers. Confirms the
    fix is entirely LLM-decoder/read-side, consistent with the substitution-not-hearing
    mechanism check. Written up in RESEARCH_PLAN.md §12.6.
10. ~~Decide + write up the pitch-representation recommendation~~ **DONE 2026-07-31** —
    RESEARCH_PLAN.md §12.6 (new section) walks decision 11's 5 requirements against the
    actual Track C–F results and states the recommendation (F0-as-text for relative pitch,
    reference-anchored image for absolute tuning, no raw learned fusion at this data scale).
11. ~~Track F aug's leakage bug~~ **DONE 2026-08-05** (commits `8050378`+`b54c4bf`, Sethu) —
    `generate_aug.py` now caps training pitches at `MAX_TRAIN_MIDI = 70`, reran 3 clean
    seeds. Corrected audio-only-on-held-out: cents 0.74 (clean, was 0.89 leaky), tuning
    0.62 (clean, was 0.83 leaky) — see Status table. First commit (`8050378`) still had a
    stale leaky seed-2 response file (analyzed before that seed's clean eval finished);
    `b54c4bf` fixed it. Fusion-null verdict unchanged.
12. ~~Tracks G/H set up, not yet run~~ **DONE 2026-08-05** (commit `8050378`, Sethu) — both
    null. See Status table for numbers and PAPER.md Results/Conclusions for the full
    writeup. Narrows the D-zoom reference-line finding to "visual + pitch-specific,"
    not a general reference-giving or chart-giving principle.
13. **Corrected 2026-08-05 (was wrong in the first pass of this entry)**: the L2 own-encoder
    re-probe for `key_id`/`mode_id`/`chord_quality`/`interval_id` was already done on
    2026-07-24 (next action 3, `results/trackB/probes/probe__*_own__{task}__*.csv`) — no
    need to re-run it. Reading the actual numbers (best layer, full label-space, not the
    4-way MCQ subset): `key_id` best 0.17–0.23 vs chance 0.042 (~5x, the strongest signal
    of the four); `chord_quality` best 0.18–0.31 vs chance 0.125 (~2x); `interval_id` best
    0.16–0.18 vs chance 0.083 (~2x); `mode_id` best 0.04–0.12 vs chance 0.077 (barely above
    chance across every encoder — the weakest signal, closest to a true floor case). So
    3 of the 4 tasks have modest-but-real recoverable signal (not near-ceiling like
    `instrument_id`'s 94/91%, but not absent either) — `mode_id` is the one where a
    front-end may genuinely have nothing to work with.
    **Superseded 2026-08-05** — replaced with a fixed sequential pipeline (Rupali's call):
    peak-picked chroma → zoomed peak-picked chroma → line graph → zoomed line graph →
    piano-roll → tonal centroid, same discipline as C–F throughout (dropout training,
    wrong-condition mechanism controls, held-out splits, 3-seed paired McNemar). **Policy
    updated 2026-08-05 (second call)**: run the FULL sequence rather than stopping at the
    first win — the goal now includes "what works best / creates the richest internal
    representation," a comparison question the stop-early rule can't answer. Train each in
    order (still sequential so later steps can build on what earlier ones show), analyze
    all six once landed. Tracked as tasks (see task list, Tracks L/M/N/O/P/Q):
    - **Track L — peak-picked (binarized) chroma**: threshold Track G's `chroma_cqt` output
      (top-K active bins per frame → bright block, rest dark) instead of the raw continuous
      energy heatmap. Cheapest fix: overtones/timbre bleed energy into neighboring bins in
      the raw chroma, making it genuinely blurry even for clean notes; this removes that
      noise without needing full note transcription.
    - **Track M — zoomed peak-picked chroma**: same binarized chroma, higher resolution /
      stretched time axis — isolates the "zoom" half of D-zoom that Track G's flat chroma
      never tested (Track G already covered the "force" half; see corrected note above).
    - **Track N — line graph (multi-pitch trajectory)**: generalizes the F0-contour that
      worked for pitch to polyphonic content — audio-derived (not MIDI-derived) pitch
      trajectories per detected note, multiple simultaneous lines during a chord. Needs a
      feasibility check first: block-chord stimuli (near-simultaneous onsets) may not suit
      a "trajectory" framing as well as more sequential content (mode/interval melodic
      forms) — may need per-task handling.
    - **Track O — zoomed line graph**: same representation, zoomed, mirroring D-zoom's
      exact recipe generalized to multiple lines.
    - **Track P — piano-roll**: absolute pitch height × time, one block per detected note
      (duration included, audio-derived onset+pitch detection). Directly targets the
      simultaneity bottleneck TASKS.md already flagged for `chord_quality`/`interval_id`
      (arpeggiated-succeeds/block-fails) — chords show as vertically-stacked blocks at one
      time-coordinate, which neither chroma (sums into 12 bins) nor a single line can show.
      Last step in the sequence; if this is also null, the input-representation search for
      harmony is exhausted for now — the next lever is the auxiliary self-transcription
      training objective (§12.3), not another front-end.
    Explicitly ruled out this round (Rupali's calls, not just mine): a text-based reference
    hint (would repeat Track E's already-diagnosed substitution-not-hearing pattern without
    teaching the model to listen better, and the "give it a reference" idea already failed
    once in-audio via Track H); resynthesizing the input audio to remove complexity (not a
    deployable fix — users' music can't be simplified before asking the question); an
    MCQ-template-glyph image (assumes MCQ framing, which conflicts with the longer-term goal
    of open-ended, non-MCQ questions).
    CPU-side groundwork (rendering, manifests, held-out splits) can be done on the laptop,
    same as Track G/H; GPU training/eval steps still need the H100 box.
14. **New (2026-08-05)**: analogous representation sequence for the RHYTHM cluster
    (`tempo_bpm`, `beats_per_bar` — the two rhythm tasks in the frozen v1 battery;
    `grouping_3v6`, TASKS.md 2.7, is speced but not yet built). Unlike harmony, this
    cluster has had **zero** causal fine-tuning of any kind before now — no LoRA-only arm,
    no front-end. Same six-step ladder, mapped onto the rhythm-appropriate DSP analogues,
    same discipline as C–P (dropout training, wrong-condition mechanism controls, held-out
    splits, 3-seed paired McNemar), run in full (not stop-early, same policy as harmony's
    second call above). Tracked as tasks (Tracks R/S/T/U/V/W):
    - **Track R — tempogram** (chroma equivalent): `librosa.feature.tempogram` /
      `fourier_tempogram`, a periodicity-vs-time heatmap, same non-leakage rule as chroma
      (fixed BPM/lag axis, not tied to this stimulus's actual tempo/meter label).
    - **Track S — peak-picked tempogram**: threshold to the top-K periodicity peaks per
      frame, same fix as Track L applied to Track R.
    - **Track T — onset-strength line graph** (F0-contour equivalent): a single onset-
      envelope curve over time (`librosa.onset.onset_strength`) — literally the rhythm
      analogue of the pitch-contour line graph that worked for pitch.
    - **Track U — zoomed onset-strength line graph**: same, higher temporal resolution.
    - **Track V — beat/onset grid ("rhythm-roll", piano-roll equivalent)**: onset markers
      plotted against a metrical grid inferred from the audio's own detected tempo
      (subdivision lines from the SAME estimate the model would have access to, not from
      the ground-truth beats-per-bar label) — shows precisely when onsets land relative to
      a pulse grid without pre-supposing how many beats divide it.
    - **Track W — rhythm necklace / circular polygon** (tonal-centroid equivalent):
      audio-derived onset times folded modulo one estimated cycle length, plotted as dots
      on a circle (Toussaint, *The Geometry of Musical Rhythm* — "rhythm necklace"
      representations; onset patterns as convex polygons on a circle reveal evenness/
      symmetry properties). **Leakage care needed**: the circle's circumference must come
      from a detected periodicity (audio-derived), not from the ground-truth beats-per-bar
      count, or the number of dots the model sees would hand over the answer directly —
      same discipline as Track V's grid.
    CPU-side groundwork can start on the laptop now (all of librosa's tempogram/onset/
    onset_strength functions are already available in the venv, same as the tonnetz
    prototype above); GPU steps need the H100 box.
    **Built + tested 2026-08-05**: all 6 renderers written (`scripts/render_rhythm_repr.py`),
    verified on real stimuli, 3 real bugs caught and fixed during testing — `_reject_harmonics`-
    equivalent issue N/A here, but (1) the necklace's cycle-length scorer originally used
    unweighted onset timing, which can't distinguish n=3 from n=6 on a perfectly regular
    click train (a uniform train folded mod ANY integer multiple looks equally concentrated)
    — fixed by weighting the circular-concentration statistic by onset strength, so real
    accents (not just regular clicks) determine the cycle length; (2) the period estimator
    (envelope autocorrelation) regularly locked onto a 2x/3x sub-harmonic of the true click
    rate — replaced with the median inter-onset interval from librosa's own onset detector,
    which is far more direct and doesn't have the sub-harmonic failure mode; (3) a
    `numpy.ptp()` API break (removed from ndarray in this numpy version). Verified 12/15
    (80%) correct cycle-length detection across all 5 beats_per_bar categories (3/4/5/6/7)
    after the fixes. **Known limitation** (found by independent leakage-review pass,
    2026-08-05): Tracks V/W's `wrong_image` mechanism control draws from the whole battery
    (`image_jobs.py`'s uniform-draw design), and ~28% of a random battery sample (56/200)
    produce a near-blank rhythm-roll/necklace image (fewer than 2 onsets detected — mostly
    single sustained-tone stimuli from unrelated tasks, not click-based). This does **not**
    affect the primary image-vs-no_image analysis (verified: 0/160 of the rhythm cluster's
    OWN stimuli are sparse, they're all click tracks) — only dilutes the wrong_image
    control's statistical power for these two tracks specifically. Not fixed; flagging so
    Track V/W's wrong-image numbers get read with that caveat, not trusted at face value.
15a. **CRITICAL BUG found + fixed 2026-08-05, before ever reaching the GPU box** (independent
    correctness-review pass, confirmed separately by a direct local run of `split()` against
    the real rhythm-task jobs): `gpu/image_track_common.py` originally reused
    `train_track_c._held_out_mask` unchanged for every track's held-out split, same as
    Track G did. That function's fallback chain is soundfont -> top-quintile-`base_midi`;
    `tempo_bpm`/`beats_per_bar`'s factors are `{bpm, meter, n_bars[, beats]}` -- **neither**
    key present, so the fallback's `base_midi >= quantile(0.8)` comparison is `NaN >= NaN`,
    False for every row. Result: `_held_out_mask` silently returned all-False for Tracks
    R-W -- 0 rows held out, meaning training would have run on 100% of the rhythm data and
    `evaluate()` would write a 0-row `responses__*.parquet`, which `analyze_track_repr.py`
    would then crash reading back (`KeyError` on missing columns). This would only have
    surfaced AFTER a full remote GPU training run for all 6 rhythm tracks -- an expensive
    way to discover a data-splitting bug. Confirmed via direct test: `split()` on the real
    jobs returned `held=0` for every one of R/S/T/U/V/W before the fix.
    **Fix**: added a third fallback tier in `gpu/image_track_common._held_out_mask` (NOT
    modifying `train_track_c._held_out_mask` itself -- that function is already used by
    Track C/D/G/H's already-run results; changing it risks changing their behavior on any
    rerun) -- rows with neither soundfont nor base_midi now fall back to a held-out
    top-quintile-BPM split, same "hold out the tail of a continuous factor" discipline as
    the base_midi tier, guarding against tempo-memorization instead of pitch-memorization.
    Re-verified after the fix: all 12 tracks (L-Q AND R-W) now produce non-empty,
    non-overlapping train/held splits with every target task represented in both (L-O:
    287 train / 612 held, matching Track G's already-published split exactly; R-W: 127
    train / 132 held).
15b. **Second bug found + fixed 2026-08-05** (same independent review): `_peak_pick`
    (Tracks L/M, `scripts/render_harmony_repr.py`) and the tempogram-picking logic (Track S,
    `scripts/render_rhythm_repr.py`) used plain `argsort`-based top-k selection with no
    energy floor -- confirmed via testing that a fully silent frame (all-zero chroma/
    tempogram) still gets exactly k bins marked "on" (argsort of an all-zero/tied array
    still returns k indices). Any stimulus with lead-in/trailing silence or a rest gap would
    get fabricated "active pitch class"/"active periodicity" markings on frames with no real
    content. Fixed: both now gate on frame energy (silent/near-silent frames, <2% of the
    stimulus's peak frame energy, are left fully off) before picking top-k on the rest.
15c. **New (2026-08-05)**: auxiliary self-transcription training objective (RESEARCH_PLAN.md
    §12.3, specced since before Track D but never run) — extend to run across **all three**
    clusters (pitch, harmony, rhythm), not harmony alone. Blocked on one open question
    first: the transcription format (`RUPALI_READ_THIS.md` §5) — plain MIDI-as-text is
    ruled out (fails requirement 5, no continuous pitch/microtiming), candidates are a
    JSON-ish event list, prose description, or a compact onset/pitch-contour grid. Resolving
    this is the actual unblock; it hasn't been picked yet. This is the one intervention in
    the whole project where the model generates its own intermediate representation during
    training rather than reading an externally-injected one — the framing every front-end
    track above is really scaffolding toward, not a replacement for Tracks L–W.

## Known gaps / honesty list
- L1 key detection weak on minor progressions (naive Krumhansl) — use
  essentia on GPU box before making L1 claims about cadence-form stimuli.
- Mode melodies are random diatonic walks — musician-composed melodies
  would be better stimuli (planned).
- Real Gemini/OpenAI payload paths and Qwen loading only testable on the
  GPU box (self-test covers everything else).
- Explain-format responses can be confabulated — evidence, not proof.
- progression_id has only 32 stimuli (small N — hypothesis-generating only).
- Contamination: synthetic stimuli are safe, but Tier 3 datasets are public.
- **key_id MCQ distractors are circle-of-fifths neighbors + the relative key by design**
  (`musicprobe/prompts.py`) — MCQ confusion matrices for this task will always look
  "musically structured" regardless of whether the model is really listening. Only the
  open-format subset is a valid listening-structure signal, and n=20/model there is small.
- MCQ vs open-format accuracy gaps are large and inconsistent across models/tasks (not just
  a minor scoring nuance) — e.g. Gemini scores 0/20 open-format on `key_id` despite ~42%
  MCQ; `progression_id` is 0% open-format for 4/6 models. Any task/model accuracy number
  should be read alongside its MCQ-vs-open split before being quoted, not just the pooled
  accuracy.
- `beats_per_bar`'s audio_gain is negative for 4/6 models AND its wrong-audio-control delta
  is negative on average (models do *better* with swapped-in wrong audio than the correct
  clip) — the only task where this happens. Treat this as a task-validity flag, not a
  capability finding, until it's manually audited (next action #3).
- ~~The Track B attention diagnostic on the 4 newer models is RETRACTED~~ **RESOLVED
  2026-07-25** (commit c348ea6): re-run with `assert_eager_attention()` passing for every
  model (top-level + every sub-config resolved to `'eager'` — the 2026-07-24 sdpa-fallback
  risk did not recur). Corrected findings, which **supersede** the retracted run and are now
  safe to cite:
  - No decay across generation steps — the retracted run's "listens first, then coasts" was
    itself an artifact of the unverified run, not a real pattern.
  - Every model attends to audio tokens *below* the uniform baseline (audio is ~55–67% of
    context; actual attention is 0.03–0.31) — all models systematically under-weight audio
    relative to its share of the input, to very different degrees.
  - Real cross-architecture variation, confirmed this time: Qwen2-Audio highest (~0.31,
    structured peak in early-mid layers), Music-Flamingo/AF3 ~0.08, Qwen2.5-Omni ~0.045,
    Qwen3-Omni-30B lowest (~0.03). Qualitatively similar ordering to the retracted run, but
    now trustworthy — graph at
    `results/trackB/attention/attention_graph.png` (`gpu/plot_attention.py`).
- No-audio "refusal" behavior is not comparable across models as a single number: Gemini
  refuses explicitly (57%), Qwen2-Audio refuses some (13%), but AF3/Music-Flamingo/
  Qwen2.5-Omni/Qwen3-Omni never refuse (0%) — they answer confidently and wrong instead,
  which is a worse failure mode than refusing, not a better one, even though 0% refusal
  reads as "more helpful" at a glance.
