# Benchmark landscape — datasets, published scores, training data, and why we built our own

Compiled 2026-08-13 via literature survey (two research passes, arXiv IDs spot-checked
against `export.arxiv.org` — all resolve to the claimed papers). This is the answer to
four standing questions: what datasets exist to test music LALMs on, how published models
score on them, what those models were actually *trained* on (requested to be comprehensive),
and — given all of that — what we're actually looking for in a benchmark that these don't
give us. Companion to `RESEARCH_PLAN.md` §1.6-1.7 (which this supersedes/expands) and
`PROJECT_STATE.md` next actions 18 and 23 (dataset hunt, training-data desk research).

Citation convention below: **(own)** = claimed in that model's own paper/report.
**(3rd)** = a *different* paper or leaderboard evaluated that model — several of the
highest-profile numbers in this space (Qwen2.5-Omni's and Gemini's MTG/GTZAN scores,
GPT-4o-Audio's numbers almost everywhere) are competitor labs' own re-runs, not
vendor-reported. Treat those as "as reproduced by a rival lab," not ground truth.
"Not found" means searched and not located — not fabricated.

---

## 1. Datasets & benchmarks for evaluating music LALMs

| Dataset/Benchmark | Task type | Size | Chat-ready vs. raw MIR | Access status | Contamination risk | Source |
|---|---|---|---|---|---|---|
| **MuChoMusic** (ISMIR'24) | MCQ knowledge+reasoning | 1,187 MCQs / 644 tracks | Ready-made | Open (GitHub+HF) | High — audio drawn from MusicCaps (417) + Song Describer (227); paper itself proved text-only LLMs hit 51–56% via priors | 2408.01337 |
| **RU-MuChoMusic** ("Are you really listening?", ISMIR'25) | Filtered/robustified MuChoMusic | Subset | Ready-made, stricter drop-in replacement | Open | Same audio, but filters distractors via a "Perceptual Index" to fight text-prior exploitability | 2504.00369 |
| **MMAU** (ICLR'25) | 27 tasks, info-retrieval + reasoning, sound/speech/music | 10,000 clips | Ready-made | Open (GitHub) | Moderate-high, web-sourced | 2410.19168 |
| **MMAU-Pro** (Aug'25) | 49 skills, expert QA, multi-audio/spatial | 5,305 instances | Ready-made | Open, live leaderboard | Lower (newer, curated) | 2508.13992 |
| **CMI-Bench** (ISMIR'25) | 14 classic MIR tasks reframed as instructions, real MIR metrics | 20 underlying datasets | Instruction-format adaptation + eval toolkit | Open (GitHub+HF) | **Explicitly documented**: attributes Qwen2-Audio's outsized MTG-Jamendo/FMA scores to training-data overlap | 2506.12285 |
| **MARBLE** | 18 tasks / 12 datasets, encoder-probing | 12 datasets | Raw MIR/probing, not chat-QA | Open | N/A (probing framework) | 2306.10548 |
| **AIR-Bench** | GPT-4-graded open chat QA, speech/sound/music/mixed | ~2.2K music chat + 19K foundation MCQ | Ready-made; music is 1 of 4 dims, not dedicated | Open | Moderate | 2402.07729 |
| **MMAR** (NeurIPS'25) | Deep reasoning, 4-layer taxonomy (Signal→Cultural), CoT rationales | 1,000 QA triplets | Ready-made | Open | Lower (curated 2025) | 2505.13032 |
| **PitchBench** (2026) | Pitch perception, 28 controlled experiments (absolute/relative, chords, loudness/duration/source/time-stretch/noise confounds) | 28 experiments | Ready-made — **closest published neighbor to our own battery design** | Open | Low (synthetic/controlled) | 2605.26176 |
| **HumMusQA** (2026) | Expert hand-written QA, perception + unimodal-shortcut robustness | 320 questions | Ready-made, small | Open | Low | 2603.27877 |
| **MUSE Benchmark** (2025) | 10 tasks: beginner (instrument/melody-shape/oddball/rhythm/pitch-shift) + advanced (chord/key-modulation/chord-sequence/syncopation/meter), N=200 human baseline | 10 tasks | Ready-made | Open (GitHub) | Low | 2510.19055 |
| Core Music Perception Tasks (2025) | Syncopation, transposition detection, chord-quality; audio-vs-MIDI, CoT/LogicLM | 3 tasks | Eval methodology, not a released leaderboard | Unclear | Low | 2510.22455 |
| Factual/Musical Eval Metrics (2025) | 6 factual-IR tasks (precision/recall/F1), reframes MusicQA-style eval for factual correctness | 3 underlying datasets (MusicNet, FMA, OverClocked ReMix) | Eval protocol | New | Substrate datasets are old/public | 2511.05550 |
| **GTZAN** | Genre classification | 1,000 clips / 10 genres | Raw MIR, widely reused as LALM probe | Public since 2002; downloads sometimes break | **Very high** — 24 yrs old, ubiquitous; documented 93 mislabelings persist in most splits | Sturm 2013 (1306.1461) |
| **MTG-Jamendo** | Multi-label tagging (genre/mood/instrument) | ~55K tracks | Raw MIR, reused directly | Open | **High** — CMI-Bench names this as the Qwen2-Audio contamination vector | 2506.12285 |
| **FMA** | Genre/metadata | 106K tracks (full) | Raw MIR/archive | Open | **High** — same CMI-Bench finding | 2506.12285 |
| **MusicCaps** | Captioning/retrieval | 5,521 clips | Raw MIR; also the literal audio substrate of MuChoMusic and MU-LLaMA's training set | Open (YouTube-ID-based; links rot) | **High**, and structurally entangled — used as *training* data (MU-LLaMA) and *eval* audio (MuChoMusic) at once | 2311.10057 (context) |
| **NSynth** | Isolated-note instrument classification/pitch | 305,979 notes | Raw MIR, reused in AF2/AF3/Music Flamingo evals | Open, well-hosted (Magenta) | Moderate-high (old, heavily reused in pretraining) | 2507.08128 |
| **GiantSteps Key** | Key detection, real EDM | 604 tracks | Raw MIR | Annotations fine; **audio scraped from Beatport preview URLs — hosting known to rot** (confirmed dead 2026-08-12, next action 23) | Low-moderate in principle, but **practically unusable** | github.com/GiantSteps |
| **GiantSteps Tempo** | Tempo estimation | Companion set | Raw MIR, reused directly (Qwen2.5-Omni reports 0.88 on it) | Same Beatport dependency | Same rot risk | 2503.20215 |
| **MedleyDB v1/v2** | Multitrack, melody F0, instrument activation | 122–196 multitracks | Raw MIR | **Audio gated behind Zenodo access request**; access requested 2026-08-12, pending (next action 23) | Moderate (gated → lower general contamination, but real access friction) | Zenodo 1649325 |
| **VocalSet** | Vocal technique classification | 10.1 hrs / 20 singers | Raw MIR | Open, direct Zenodo download, no gating | Moderate | Zenodo 1442513 |
| **GuitarSet** | Guitar transcription/technique | Preview + full | Raw MIR | Open (UCSD/Zenodo) | Moderate | ISMIR 2018 |
| **IDMT-SMT-Guitar** | Guitar playing-technique classification | 4,700+ single notes | Raw MIR | Open, but **CC BY-NC-ND** (no derivatives/commercial reuse) | Moderate | Zenodo 7544110 |
| **Harmonix Set** | Beat/downbeat/structure annotation | 912 tracks | Raw MIR | Annotations+melspecs open; **raw audio NOT distributed** (copyright), must source separately via MusicBrainz IDs | Moderate | github.com/urinieto/harmonixset |
| **SALAMI** | Hierarchical song-structure annotation | 1,359 tracks | Raw MIR | Same audio-absent pattern as Harmonix (not independently re-confirmed) | Moderate | — |
| **RWC** | Multi-genre reference collection | Varies | Raw MIR | Gated, originally physical-media distribution | Low leakage risk, poor usability | — |
| **MusicNet** | Classical multitrack, note-level annotation | 330 recordings | Raw MIR | Open | High (old, standard MIR training set) | 2511.05550 |
| **POP909** | Chinese pop, MIDI+audio | 909 songs | Raw MIR, symbolic-leaning | Open (GitHub) | Moderate | 2008.07142 |
| **Slakh2100** | Multitrack synthesized instrument mixes | 2,100 tracks | Raw MIR (source separation) | Open | Moderate (synthetic but public) | slakh.com |
| **OpenMU-Bench** | Captioning, reasoning, MCQ, lyrics, tool-use | ~1M examples | Training-data bench, not eval-only | Open | Bootstrapped via GPT-3.5 + existing datasets | 2410.15573 |
| **Song Describer Dataset** | Free-text captioning/retrieval | 1.1K captions / 706 tracks | Ready-made; also feeds 227 MuChoMusic tracks | Open, CC-licensed | Lower than MusicCaps by design, but overlaps MuChoMusic | 2311.10057 |
| **JamendoMaxCaps** | Large-scale captions w/ imputed metadata | 362K instrumental tracks | Captioning-scale, not chat-QA | Open (CC) | Newer (2025), lower risk, but same MTG-Jamendo family | 2502.07461 |
| **MusICA-MetaBench** ("Music I Care About", 2026) | On-demand meta-benchmark: auto-derives MCQ from *any* audio you feed it — pitch, interval, rhythmic notation, temporal proportion, harmonic analysis, piece-level | Framework, not fixed size (bundles ChoraleBricks real + ChoralSynth synthetic) | Ready-made, but **MCQ only** (5-option incl. "none correct") | Open (framework) | Low on its own bundled data (ChoraleBricks); depends entirely on what you feed it if reused generically | 2607.06015 |
| **BASS** (2026) | 12 tasks / 4 categories: structural segmentation, lyric transcription, artist collaboration, and genre (**"musicological analysis" = genre tasks only, despite the name — no key/chord/harmony content**) | 2,658 questions / 1,993 unique real songs / 138 hrs | Ready-made | Open (GitHub+HF) | Not assessed directly, but its own headline finding is a text-prior one: metadata-only (artist+title, no audio) *improves* lyric-transcription score over audio-only | 2602.04085 |
| SongBench (2026) | **Not a comprehension benchmark** — rates AI-*generated* song quality across 7 dimensions (vocal/instrument/melody/structure/arrangement/mixing/musicality) against expert ratings | 11,717 expert-rated generated samples | Generation-quality eval, out of scope for LALM-comprehension testing | Open (GitHub, Tencent) | N/A (evaluates generated audio, not a training-data-contamination-relevant corpus) | 2604.25937 |
| SongEval (2026) | **Not a comprehension benchmark** — aesthetic ratings (coherence, memorability, vocal naturalness, structure clarity, musicality) on full songs, for judging generative/enhancement models | 2,399 full songs / 140+ hrs, EN+ZH | Generation-quality eval, out of scope | Open (HF: ASLP-lab/SongEval) | N/A, same reason as SongBench | 2505.10793 |

**Read on this table**: everything genuinely "chat-ready" (MuChoMusic, MMAU family, AIR-Bench,
MMAR, CMI-Bench) either tests caption/tag-grade semantics or is only 1-2 years old and
already shows measurable contamination (CMI-Bench's own finding). Everything with
note/interval/cents-grade precision is either raw MIR requiring adaptation, or has rotted
access (GiantSteps), or doesn't ship audio at all (Harmonix, SALAMI). The two closest
published neighbors to this project's own approach are **PitchBench** (2605.26176, pure
pitch psychophysics, 28 controlled experiments) and **MUSE Benchmark** (2510.19055, human
baseline + beginner/advanced tiering) — worth reading closely for design overlap/contrast
before finalizing the paper's related-work framing.

---

## 2. Published model performance

### Qwen2-Audio-7B
- MMAU overall: **52.50%** (3rd, 2410.19168 baseline table)
- MuChoMusic: "Qwen-Audio" v1 **51.4%** (3rd, 2408.01337); AF3's paper separately reports
  "Qwen2-Audio-Instruct" **46.2%** (3rd, 2507.08128) — different generations, don't conflate.
- AIR-Bench chat, Music dim: **6.79/10**, best of models AIR-Bench tested (3rd, 2402.07729)
- CMI-Bench (3rd, 2506.12285): GTZAN genre 72.07% (SOTA 83.9), GiantSteps key 8.28 (SOTA
  74.3), beat F-measure 23.69 (SOTA 88.3), melody-extraction 5.06% (SOTA 72.3), lyrics WER
  115.7 (SOTA 12.99, lower better). **CMI-Bench explicitly attributes the comparatively
  strong genre/tagging numbers to training-data contamination, not generalized skill.**

### Qwen2.5-Omni
- MMAU (own, 2503.20215 Table 3): Sound 67.87%, **Music 69.16%**, Speech 59.76%, Avg
  65.60% — own-reported baselines Gemini-1.5-Pro 54.90%, Qwen2-Audio 49.20%.
- MMAU-Pro leaderboard (3rd): Avg 52.2, Music 61.5.
- GiantSteps Tempo (own): 0.88 vs LLark-7B 0.86.
- MMAR Music: 46.12% (3rd, via Music Flamingo's table, 2511.10289).
- MUSE Benchmark: at/near chance on advanced tasks (chord/key-modulation/syncopation)
  (3rd, 2510.19055).

### Qwen3-Omni
- Own paper (2509.17765 Table 8, "RUL-MuchoMusic" = RU-filtered variant): **52.0** vs.
  Gemini-2.5-Pro 49.4, GPT-4o-Audio 36.1 (latter two run by the Qwen team, not vendor-self-
  reported). Same table: GTZAN 93.0%, MTG-Genre 39.0 Micro-F1, MTG-Mood/Theme 21.0,
  MTG-Instrument 40.5, MTG-Top50 36.7, MagnaTagATune 44.3 — reported beating both
  competitors on every row.
- MuChoMusic: 52.10% (3rd, via Music Flamingo's comparison table).
- MusicCaps (GPT-graded): 7.2 (3rd) vs. Music Flamingo's 8.8.

### SALMONN (original)
- No clean own-paper scalar music metric located (qualitative claims only in accessible
  text) — flag for a direct full-PDF check if a number is needed.
- MuChoMusic: 41.8% (3rd, 2408.01337). AIR-Bench Music: 5.95/10 (3rd, 2402.07729).
- MMAU-Pro leaderboard (3rd): SALMONN-7B Avg 34.5/Music 44.9; SALMONN-13B Avg 39.6/Music 47.2.

### SALMONN-2 (2607.17079)
- Own paper: MMAU-Pro 58.5, MMAR 64.5, MMSU 69.5 — overall scores, not confirmed
  music-isolated; claims SOTA among comparable-scale open models.

### Audio Flamingo 2 (2503.03983)
- Own: MMAU Music 72.9%; MuChoMusic 56.5%; Music Instruct Long 90.2%; MusicQA 93.0%
  (vs. prior SOTA MusiLingo 90.0%).
- MMAU-Pro leaderboard (3rd): Avg 42.6, Music 55.7.

### Audio Flamingo 3 (2507.08128)
- Own, Table 2: MMAU Music 74.47% (+think 74.60%) vs. Qwen2.5-Omni 67.33% (AF3's own
  re-run); MusicAVQA 76.7% vs. 73.4%; Music Instruct Long 92.7%; MuChoMusic 47.4% (+think)
  vs. Qwen2-Audio-Instruct 46.2%; NSynth 78.9%. (A separately-cited AF3 MMAU-overall figure
  of 72.42/Music 73.9 appears elsewhere in the paper — likely a different checkpoint/table
  version; flagged as an internal inconsistency, not resolved here.)
- MMAU-Pro leaderboard (3rd): Avg 51.7, Music 61.7.

### Music Flamingo (NVIDIA, 2511.10289) — all own-paper; every "competitor" number is Music
Flamingo's own re-run of that competitor, not vendor-self-reported
- MMAU Music **76.83%** vs. AF3 73.95% · MMAU-Pro Music **65.60%** vs. Gemini 2.5 Flash
  64.90% · MuChoMusic **74.58%** vs. Qwen3-Omni 52.10% · MMAR Music **48.66%** vs.
  Qwen2.5-Omni 46.12% · Music Instruct (GPT-graded) **97.1** vs. AF3 92.7 · NSynth
  instrument **80.76%** vs. AF2 78.9% · GTZAN genre **84.45%** vs. Pengi 80.00% ·
  Medley-Solos-DB **90.86%** vs. AF2 85.80% · MusicCaps (GPT-graded) **8.8** vs. Qwen3-Omni
  7.2 · Lyrics WER Opencpop **12.9** vs. GPT-4o-Audio 53.7, Qwen2.5-Omni 55.7 (lower better)
  · Lyrics WER MUSDB18 **19.6** vs. GPT-4o-Audio 32.7, Qwen2.5-Omni 68.7 · SongCaps
  (human-rated /10) **8.3** vs. AF3 6.5.
- **No model's own paper reports a MuChoMusic score above ~57% until this 74.58% figure** —
  and it hasn't yet been independently reproduced by a third party.

### Gemini 2.x/3.x native audio
- MMAU-Pro leaderboard (3rd): Gemini-2.0-Flash Avg 55.7/Music 56.9; Gemini-2.5-Flash Avg
  59.2/Music 64.9; later rows (Gemini-3.1-Pro-Preview, 3.5-Flash, 3.6-Flash, current as of
  Aug 2026) at avg 68–72%/music 70–76% — leaderboard is continuously updated, not a frozen
  paper snapshot, so re-check the live JSON before citing a specific number.
- Qwen3-Omni's own table (3rd re-run of Gemini-2.5-Pro): RUL-MuchoMusic 49.4, GTZAN 81.0,
  MTG-Genre 32.5, MTG-Mood 8.9, MTG-Instrument 22.6, MTG-Top50 21.6, MagnaTagATune 30.1 —
  lower than Qwen3-Omni's own scores everywhere, but from a competitor's paper.
- A widely-repeated "Gemini 2.5 Pro MMAU Avg 71.6" figure traces to a leaderboard
  aggregator (llm-stats.com), likely originating in Step-Audio 2 (2507.16632) —
  **low-confidence provenance, verify against that PDF directly before citing.**
- **No Google-authored report with Gemini's own claimed music-benchmark numbers found —
  every Gemini music number above is third-party.**

### Music-specific models (frozen-MERT family)
- **MU-LLaMA**: MuChoMusic 32.4% (3rd, 2408.01337) — second-worst of 5 models MuChoMusic
  tested. Own-paper MusicQA numbers not independently re-verified; AF2's paper treats
  MusiLingo, not MU-LLaMA, as the prior MusicQA reference point.
- **MusiLingo**: MuChoMusic **21.1%** (3rd) — **worst of the 5 models tested**, despite
  being purpose-built for music. MusicQA 90.0% (3rd, cited as prior SOTA before AF2's 93.0%).
- **M2UGen/MuMu-LLaMA**: MuChoMusic 42.9% (3rd), with the highest instruction-following
  rate (96.4%) among the smaller specialist models. Own-paper SOTA claims across 4 tasks
  not independently re-extracted here.

### Cross-cutting flags
- **CMI-Bench is the sharpest documented contamination finding in this whole survey** — it
  names the mechanism (Qwen2-Audio's MTG/FMA training overlap), not just gestures at risk.
- **MusicCaps is structurally double-booked**: simultaneously common training data
  (MU-LLaMA) and the literal eval audio of MuChoMusic — a train/eval leakage vector baked
  into the ecosystem, not a one-off hygiene issue.
- **"Are you really listening?" (2504.00369)** independently reproduces this project's
  central finding at benchmark scale: text-only LLMs (zero audio) hit up to 56.4% on
  original MuChoMusic, and audio-conditioned models still score above chance with the audio
  replaced by Gaussian noise.
- **Music is the weakest or near-weakest domain on nearly every general audio-reasoning
  benchmark** — MMAU-Pro's own human baseline for music (70.5%) is the *lowest human
  score of any domain*, suggesting the difficulty is partly in task design, not only model
  capability.

---

## 3. What these models are actually trained on (comprehensive, per request)

The consistent finding: **general-purpose LALMs treat music as a rider, not a target.**
Every dedicated music-specialist model (Music Flamingo, MU-LLaMA, MusiLingo, M2UGen) makes
music-specific training data legible and named; every general-purpose model (Qwen family,
Gemini) does not name a single music-specific training dataset — music enters only as an
*evaluation* target inside a broader audio-understanding benchmark (AIR-Bench, MMAU,
GTZAN/MTG/MagnaTagATune). This directly supports the project's H3/H4-adjacent framing:
poor music performance may reflect what these models were optimized for, not only encoder
limits.

| Model | Music-specific pretraining data? | Music-specific SFT data? | Explicit music curriculum/reward? | Data-mix % disclosed? |
|---|---|---|---|---|
| **Music Flamingo** | Inherits AF3 encoder pretraining (not music-dedicated at that stage) | **Yes, extensive**: MF-Skills (~5.2M: ~3.4M captions + ~1.8M QA, 4-stage pipeline incl. real MIR tools — madmom/essentia/Chordino/Parakeet); MF-Think (~176K CoT, fact-checked, discarded if >30% steps wrong); Table 2 names LP-MusicCaps MSD, MSD Captions/QA-refined, Music4All, NSynth, FMA, MusicBench, Mu-LLaMA, MusicAVQA, MusicQA | **Yes** — dedicated pipeline; GRPO with 3 custom rewards (format, accuracy, and a structured-metadata reward scored against genre/BPM/key/instruments/theory annotations). Authors explicitly rewrote prior captions to fix "mislabels of tempo, key, and timbre" and reframed MCQs "to reduce language priors" | Partial — per-dataset hours/pairs given (Table 2), no full pretraining-corpus % table |
| **Audio Flamingo 3** | Partial — AF-Whisper's 13.2M pairs / 31 datasets (Table 5) include ~1.4M music pairs (MSD/Music4All/FMA), ≈10-15% of the disclosed subset (our arithmetic, not an authors'-stated %) | Partial — AudioSkills-XL: 2M music-reasoning pairs of 8M total; **AF-Think discloses a clean 100K/50K/100K speech/sound/music split**; AF-Chat discloses 35K/40K | No dedicated stage — music is one skill category in a general curriculum | Partial — AF-Think/AF-Chat subsets disclosed, full corpus not |
| **Qwen2-Audio** | Unknown — Figure 3 (pretraining-hours chart) is an **image, not text-extractable**; confirmed unresolved on a second research pass (both PDF and HTML arXiv fetch attempted) | Not disclosed — no dataset names for SFT/DPO stages | No | **No** |
| **Qwen2.5-Omni** | Token counts only (300B audio of ~1.1T total), no content-type split | Not disclosed ("audio-modality conversation data," no names) | No — music enters only via MMAU as an **evaluation**, never training | No |
| **Qwen3-Omni** | 20M hours; disclosed split is ASR-language-based (80% Zh/En, 10% other-lang ASR, 10% "audio understanding" — itself not broken down) | Not disclosed; music eval only via GTZAN/MTG/MagnaTagATune, not training | No | Partial (language/task split only, not content-type) |
| **Qwen3.5-Omni** (2604.15804, ~Apr 2026 — newer than prior desk research, verified to exist) | 40M hours, **synthetically labeled by Qwen3-ASR** (i.e. recursive self-labeling); language ratio 3.5:3.5:3 (Zh:En:multilingual) | Not disclosed | No | Paper explicitly states it does **not** provide sound/music/speech breakdowns |
| **SALMONN** | No (pretraining = speech-ASR + general audio captioning only) | **Yes, minor**: MusicCaps (14 hrs) + MillionSong/MusicNet (403 hrs) named explicitly, ~417 of ~4,400 total instruction hours | No dedicated stage/reward, but music is a named first-class output capability (the model's name literally spells out "...Language Music...") | Yes for the instruction stage (Table 1 gives per-dataset hours) |
| **Gemini 2.x native audio** | Not disclosed (confirmed — only vague qualitative statements about "audio including speech and other audio types") | Not disclosed | No mention of music as a target anywhere in the accessible report text | **No** |
| **MU-LLaMA** | N/A (frozen MERT encoder) | **Yes**: MusicQA synthesized by **MPT-7B** from MusicCaps + MagnaTagATune tags — text-LLM-from-metadata, not signal-grounded | Music is the entire scope; no RL/CoT | N/A (single-domain model) |
| **MusiLingo** | N/A (frozen MERT) | **Yes**: MusicInstruct (27,540 pairs) generated via **GPT-4** few-shot from MusicCaps captions | Music is the entire scope; no RL/CoT | N/A |
| **M2UGen/MuMu-LLaMA** | N/A (frozen MERT/ViT/ViViT) | **Yes**: MUCaps/MUEdit/MUImage/MUVideo (~167.69 hrs total), captions generated **recursively by MU-LLaMA itself** + BLIP, plus MusicQA/Alpaca. Note: the two papers report inconsistent hour/file counts for the same-named datasets — unreconciled, flagged not resolved | Music (+ cross-modal generation) is the entire scope; no RL/CoT | N/A; internal inconsistency flagged |

**The synthetic-data-from-metadata pattern is confirmed and precisely named** for the
music-specialist family: MU-LLaMA's MusicQA (MPT-7B captioning MusicCaps/MagnaTagATune
tags) and MusiLingo's MusicInstruct (GPT-4 from MusicCaps captions) are both text-LLM
paraphrases of existing captions — the model generating the QA pair never hears the audio.
This is exactly the mechanism this project's `RESEARCH_PLAN.md` §1.4 already names as the
source of caption-grained, note-blind supervision — now confirmed at the level of specific
functions (which text-LLM, which source captions) rather than as a general claim.

**Two items resolved from PROJECT_STATE.md's open gaps (next action 18):** Qwen2-Audio's
Figure 3 remains genuinely unresolved (image-only, confirmed on retry — would need manual
visual reading of the PDF, not an automated fix). A newer Qwen3.5-Omni report was found to
exist (2604.15804) and explicitly disclaims providing exactly the content-type breakdown
this project has been looking for — useful as a citable "the field doesn't report this"
data point rather than a continuing unknown.

---

## 4. What we're looking for in a benchmark — and why these aren't enough

Four properties, cross-referenced against what's actually available above:

**(a) Ground-truth precision at the right grain.** Every chat-ready benchmark in §1
(MuChoMusic, MMAU, AIR-Bench, MMAR) is built from captions/tags/metadata — genre, mood,
instrumentation, free-text description. That's the grain the underlying audio datasets
(MusicCaps, Song Describer, MTG-Jamendo) were annotated at. None of them carry note-level,
cents-level, or beat-level ground truth, because the annotation pipeline never required a
human (or algorithm) to resolve pitch/timing precisely — it required them to describe the
vibe. CMI-Bench is the one exception that reframes classic MIR tasks with real metrics, and
it's telling that LALMs collapse hardest exactly there (key 8.28 vs. SOTA 74.3, melody
5.06% vs. 72.3). **Our battery's synthetic stimuli exist specifically to buy this precision
for free** — perfect ground truth from generation, at a grain (pitch class, cents, interval,
BPM) none of these public sets offer at scale.

**(b) Contamination control that's verifiable, not assumed.** GTZAN (24 years old),
MTG-Jamendo, FMA, and MusicCaps are all high-risk by CMI-Bench's own direct evidence (named
mechanism: Qwen2-Audio's inflated MTG/FMA scores). Worse, MusicCaps is *structurally*
double-booked as both training data and eval audio across the ecosystem — a leakage vector
no amount of held-out splitting on the eval side can fix, because the leak already happened
upstream in some other model's pretraining. A benchmark's contamination story needs to be
about the *audio*, not just the *task framing* — RU-MuChoMusic filters distractors but
reuses the same (already-public, already-scraped) tracks. **Our synthetic stimuli are
contamination-proof by construction** (rendered from soundfonts, never published before);
the real-recordings arm (next action 23) is deliberately sourced from freshly-pulled
Wikimedia Commons recordings and MedleyDB (gated, lower general-scrape exposure) rather than
GTZAN/MusicCaps precisely for this reason.

**(c) Built-in shortcut controls on every item, not just at the dataset level.**
MuChoMusic's own headline result — and its independent replication in "Are you really
listening?" (up to 56.4% audio-free, above-chance under noise-substituted audio) — is that
text-prior shortcutting is pervasive and benchmark authors have to go looking for it after
the fact. None of the surveyed benchmarks bake no-audio and wrong-audio controls into
*every* item as a design invariant; RU-MuChoMusic's filtering is the closest analogue, but
it's a post-hoc subset selection, not a per-item control pair. **This is precisely why our
battery's controls (no-audio on 30%, wrong-audio on 10%, MCQ permutation, ≥3 paraphrases,
explain-jobs) are structural, not a post-hoc audit** — `audio_gain = acc_audio −
acc_no_audio` is computable per task, per model, per condition, not just at benchmark scale.

**(d) Tiered difficulty that locates *where* a failure lives, and mechanism access to say
*why*.** Every surveyed chat benchmark reports one aggregate "music" score (or a handful of
skill categories). That tells you a model is bad at music; it doesn't tell you whether it
can't hear a pitch class, can hear pitch but not intervals, or hears intervals fine but
can't track them into a key. MARBLE probes encoder representations but never pairs that with
behavioral chat accuracy on the same stimuli — so encoder capability and LLM usage of that
capability are measured on different tasks entirely, which is exactly the alignment-gap
question (L2 probe high, behavior low) this project's framework is built to catch and which
none of the surveyed benchmarks can answer. PitchBench (28 controlled pitch experiments) and
MUSE Benchmark (beginner/advanced tiers + human baseline) are the closest published
neighbors on the *behavioral* side — worth citing as convergent evidence — but neither pairs
its behavioral tier with a matched L2 probe on the same stimuli, which is what our Tier
1→2→3 design plus Track B probing is for.

**Remaining gaps even in our own approach**, flagged rather than swept under the rug:
- **Non-Western/non-12-TET coverage is thin everywhere**, including here. GTZAN's genres,
  MTG-Jamendo's tag vocabulary, and our own key/mode/cents tasks are built on a 12-TET,
  Western-tonal frame. Music Flamingo's own paper makes the same critique of prior corpora
  ("short, Western instrumental clips") and claims a ~3M-song corrective — worth checking
  whether that corrective includes Carnatic/microtonal material specifically, which bears
  directly on this project's separate data-vs-nature question about microtonal perception.
  `cents_discrimination` (Tier 1.4) is the one task designed to be tradition-agnostic, but
  the rest of the battery (key_id, mode_id, chord_quality) inherits Western-tonal ground
  truth by construction.
- **Real-full-piece audio remains the weakest arm of our own battery**, same status as
  PROJECT_STATE.md next action 23 already tracks: GiantSteps' Beatport-hosted audio is
  confirmed dead, MAESTRO's per-file URLs 404, and MedleyDB access is still pending as of
  this writing — the interim 7-recording Wikimedia Commons set is a stopgap, not a
  replacement for a real held-out-audio arm at battery scale.
- **No benchmark surveyed here (including MARBLE) reports a training-data-mix percentage
  for its own probing targets** — meaning even the “what fraction of pretraining is music”
  question this section answers per-model can't currently be cross-validated against an
  independent measurement; it rests entirely on what each lab chose to disclose.

---

## 5. The tool-aided-model claim: do we need a new benchmark, or does the existing landscape cover it?

Separate question from §4, and answered later (2026-08-16/17). §4 justifies this project's
*existing* synthetic battery (atomic/structural precision, contamination-proof by
construction). This section is about a *different* ask that came up afterward: for the
tool-aided model this project is building, we want to claim it doesn't just improve on
narrow, hyper-specific synthetic pitch tasks (which a task-specific classifier could match)
— it generalizes to **real music, with open natural-language prompts, on genuine music
theory content (not just genre/mood/vibe), including instrument- and vocal-technique
questions.** Does that need a new custom benchmark, or does combining existing ones cover it?

**A custom "famous song" benchmark was prototyped first, then set aside.** Before landing on
the answer below, we built a small pilot (`benchmark/tier1_pilot.json`,
`benchmark/tier2_pilot.json`) testing detailed theory questions on famous commercial
recordings (Let It Be, Take Five, Clocks, Billie Jean, Clair de Lune), citation-gated against
existing published sources (Hooktheory, Wikipedia, AllMusic, music journalism — no fresh
human/expert annotation) rather than hired annotators. It worked as a methodology
demonstration — it caught its own wrong fact (a chord-progression claim that a second-source
check contradicted) and rejected two plausible-sounding fan claims about live performances on
source-quality grounds alone — but it stalled on a problem that has nothing to do with
research design: **actually running it needs the real audio bytes**, and these are
commercial copyrighted recordings we can't legally host or redistribute. Buying a handful of
tracks for personal testing was the workable path, but that only covers 5 songs, doesn't
scale, and doesn't get us anything the existing landscape doesn't already provide more
rigorously. Kept as `benchmark/*.json` + a review artifact as a methodology appendix, not
scaled into a maintained tier.

**Checked against the project's actual RQs (RESEARCH_PLAN.md), existing infrastructure
already covers most of this:**
- **RQ1 (capability map)** — the project's own synthetic battery already handles
  atomic/structural precision; nothing new needed there.
- **RQ2 (mechanism — encoder vs. LLM usage)** — already unique to this project (Track
  A behavior + Track B probing on identical stimuli). No surveyed benchmark, old or new,
  pairs behavior with representation probing this way, so no existing benchmark closes or
  opens this gap either way.
- **RQ3 (failure etiology, esp. contamination)** — the famous-song tiers were built
  specifically to test this, but **CMI-Bench already ran this experiment**, more
  rigorously and peer-reviewed: it directly attributes Qwen2-Audio's inflated
  genre/tagging scores to training-data overlap with MTG-Jamendo/FMA (§2, cross-cutting
  flags). BASS adds a second, independent contamination-adjacent finding at real scale
  (1,993 songs): giving a model just artist+title (no audio) *improves* lyric-transcription
  score over audio-only, direct evidence of text-prior reliance. Citing both is stronger
  than re-deriving the same phenomenon from 5 purchased tracks.

**Final portfolio for the tool-aided-model generalization claim** (no new benchmark built):

| Role | Benchmark | Covers |
|---|---|---|
| Anchor | **CMI-Bench** | Natural-language instruction format (not MCQ), real audio, explicitly named "vocal technique recognition" + "instrument performance technique detection" tasks, theory-precision tasks (key/pitch/melody/beat, not just genre) |
| Real + unheard | **MUSE Benchmark** | Original, never-before-released recordings (genuinely unheard, not just "less famous"); chord ID, key modulation, syncopation, meter — but **forced-choice format**, not open NL |
| Real + unheard, on-demand | **MusICA-MetaBench** | Auto-generates theory questions (pitch/interval/rhythm/harmony) from *any* fed-in audio, piece-level context — but **MCQ format**, not open NL |
| Narrow-task baseline | **PitchBench** | Synthetic, controlled pitch psychophysics — this is exactly the "hyper-specific task a classifier could match" baseline the real-music results need to beat |
| Technique depth | **VocalSet / GuitarSet** (direct, not via CMI-Bench's sampling) | Real audio, CC-licensed, no gating, if more technique coverage is wanted than CMI-Bench's built-in tasks sample |
| Supplementary | **BASS** | Real audio at genre-diverse scale (1,993 songs); the text-prior/memorization finding above. Not a theory-task source — its "musicological analysis" category is genre classification only, no harmony/key/chord/technique content despite the name |
| Out of scope | SongBench, SongEval | Score *generated* music quality/aesthetics, not model comprehension — relevant only if the tool-aided model's output (not its listening) is ever being judged |

**Honest gap that remains:** natural-language format is concentrated almost entirely in
CMI-Bench; MUSE, MusICA-MetaBench, and PitchBench are all forced-choice/MCQ. Nothing
surveyed combines real + unheard + theory-precise + open-NL in one place. If that specific
combination becomes essential to the paper's claim later, the cheapest fix is *not* reviving
the famous-song plan — it's writing an open-NL question layer on top of MUSE's or
MusICA-MetaBench's already-licensed, already-ground-truthed audio, since the hard part
(real, unheard, legally clear audio with verified answers) is already solved by those two.

Also worth being precise about: CMI-Bench's "real" audio is real, publicly-available
MIR-research tracks — not necessarily mainstream-recognizable the way the famous-song pilot's
Beatles/Michael Jackson songs were. If the paper's claim is "generalizes to real full-mix
audio, not synthetic tones," this portfolio fully supports it. If it specifically needs
"famous enough that a listener would recognize it," that's still not covered by anything
here — only by the shelved pilot.

---

## 6. Does the model we're fine-tuning already do well on this portfolio? (2026-08-19)

Mentor's framing, relayed 2026-08-19: a multi-benchmark eval is only evidence of anything if
the model being fine-tuned does *poorly* across the board first — otherwise "we ran our model
on more benchmarks" isn't a contribution. This section checks that premise directly, task by
task, against the §5 portfolio, for **Qwen2.5-Omni-7B** (the Track C–Z LoRA base / the model
actually being fine-tuned).

**MUSE Benchmark — re-derived from the authors' own logs, not a new run.** MUSE's GitHub repo
(`brandoncarone/MUSE_music_benchmark`) ships full per-question logs for every model it tested,
including Qwen2.5-Omni-7B (`Gemini_Qwen_AF_logs/`, 120 log files = 10 tasks × 2 prompt modes ×
2 stimulus groups × 3 seeds, 10 questions/log). Their own paper states this qualitatively
("at or near chance on advanced tasks"); `experiments/gpu/parse_musebench_qwen.py` parses the
raw `Evaluation: Correct/Incorrect` lines directly and checks each task's actual chance level
against its own `expected=` answer field (not assumed from the task description) — output at
`experiments/results/external_benchmarks/muse_qwen25omni.csv`:

| task | tier | acc | chance | verdict |
|---|---|---|---|---|
| chord_quality | advanced | 0.500 | 0.50 (binary maj/min) | **exactly chance** |
| syncopation | advanced | 0.500 | 0.50 (binary A/B) | **exactly chance** |
| meter_identification | advanced | 0.342 | ≤0.33 (≥3-way) | **at chance** |
| chord_progression_matching | advanced | 0.550 | 0.50 (binary) | **at chance** |
| key_modulation | advanced | 0.550 | 0.50 (binary) | **at chance** |
| contourID | beginner | 0.208 | 0.25 (4-way) | **at/below chance** |
| rhythm_matching | beginner | 0.533 | 0.50 (binary) | **at chance** |
| transposition | beginner | 0.550 | 0.50 (binary) | **at chance** |
| oddballs | beginner | 0.717 | 0.50 (binary) | above chance — real signal |
| instrumentID | beginner | 0.983 | n/a (timbre ID) | near ceiling |

**Every one of the 5 advanced (theory/relational) tasks lands within a few points of chance.**
Only instrument identification (a coarse timbre task, consistent with this project's own
`instrument_id` near-ceiling result) and oddball detection show real above-chance skill. This
is not a close call — it's the same shape as this project's own battery (near-floor on
`mode_id`/`key_id`/`interval_id`, near-ceiling on `instrument_id`/`octave_id`), now confirmed
on an independent benchmark, independent stimuli, and independent authors.

**A second, qualitative finding worth citing alongside the numbers**: several logs show the
model emitting near-identical templated reasoning text across genuinely different audio
files — e.g. `chord_progression_matching`'s COT/GroupA/seed1 log reasons "I–vi–IV–V
progression in the key of C major" for both excerpts, on every question in the log, regardless
of which two (different, non-C-major) files were actually presented. This is independent
third-party evidence of the same mechanism this project's own attention diagnostic already
quantified (`gpu/attention_audio.py`: audio tokens attended at 0.03–0.31 against a uniform
baseline of ~0.55–0.67) — the model is pattern-completing a plausible-sounding answer rather
than reading the audio, not an isolated scoring artifact.

**CMI-Bench, PitchBench, BASS — harnesses built and schema-verified 2026-08-19; GPU runs not
yet executed (no GPU on the laptop).** Unlike MUSE, none of these three have an existing
Qwen2.5-Omni log to mine, so each needed a real first-party harness rather than a re-derivation.
Full detail in `experiments/scripts/RUNBOOK_external_benchmarks.md`; short version:
- **CMI-Bench** (`gpu/eval_cmibench.py`) — this portfolio's NL-format anchor, covers
  key/melody/beat/vocal-technique/instrument-technique with published SOTA-gap context already
  in §2 (Qwen2-Audio trails specialist MIR SOTA by 60+ points on key detection and melody
  extraction). Real repo cloned and read directly to get the I/O format right — their scorer
  expects output as a **JSON array**, not JSONL, despite the `.jsonl` filename, a real trap the
  obvious assumption would have walked into silently.
- **PitchBench** (`gpu/eval_pitchbench.py`) — the closest published neighbor to this project's
  own battery (§5), confirmed live on HF (`pitchbench-authors/PitchBench`, 30 configs, CC BY
  4.0) via the `datasets-server` API directly. Their own paper tested 6 frontier models
  including Qwen-**3.5**-Omni (not 2.5) and still found "pitch hearing remains highly
  unreliable" — itself citable corroborating evidence regardless of what our own run of
  Qwen2.5-Omni shows.
- **BASS** — schema confirmed real and live, but genuinely **blocked**: each row's audio field
  is a bare filename with no resolvable path or embedded bytes alongside it, and the only
  fallback (`youtube_url`) is the same fragile path already deprioritized once for MuChoMusic's
  MusicCaps subset. Lower priority anyway (§5: supplementary, not anchor).
Not built: **GuitarSet** (no chat-QA wrapper exists anywhere, would need one built from
scratch like `real_music_medleydb.py`; VocalSet's already covered — it's one of CMI-Bench's
own sub-tasks). **MusICA-MetaBench** (no public code repo locatable at all; already
deprioritized in §5 as framework/MCQ-only).

**Reading for the mentor conversation**: the premise holds, on the evidence gathered so far.
Qwen2.5-Omni does not do well across this portfolio's theory/relational tasks — it's at chance
on structural music understanding and only "does well" on the one coarse task (timbre/
instrument ID) every model in this survey also does well on. That's consistent with, not
contradicted by, this project's own capability map. It does NOT by itself answer the second
half of the mentor's worry — that the project's contribution can't just be "a fine-tuned model
that scores higher on these same benchmarks." A model that closes this gap through more
training data on these specific task formats would be a capability result, not a music-
understanding-research contribution; what makes Tracks C–Z's findings a research contribution
is that they explain *why* the gap exists at the representation level (front-ends that fix
pitch don't transfer to harmony/rhythm; the encoder itself is frozen under LoRA; wrong-audio
controls show most gains are readout-level, not perceptual) — see RESEARCH_PLAN.md §12.6 and
decision 16 in PROJECT_STATE.md. The benchmark-gap evidence in this section is motivation for
*why the study matters*, not a substitute for that mechanistic contribution.

---

## Sources (arXiv IDs, spot-checked 2026-08-13, plus 4 more spot-checked 2026-08-17)

2408.01337 (MuChoMusic) · 2504.00369 (RU-MuChoMusic) · 2410.19168 (MMAU) · 2508.13992
(MMAU-Pro) · 2506.12285 (CMI-Bench) · 2306.10548 (MARBLE) · 2402.07729 (AIR-Bench) ·
2505.13032 (MMAR) · 2605.26176 (PitchBench) · 2603.27877 (HumMusQA) · 2510.19055 (MUSE
Benchmark) · 2510.22455 (Core Music Perception Tasks) · 2511.05550 (Factual/Musical Eval
Metrics) · 2311.10057 (Song Describer Dataset) · 2502.07461 (JamendoMaxCaps) · 2410.15573
(OpenMU) · 2308.11276 (MU-LLaMA) · 2309.08730 (MusiLingo) · 2311.11255 (M2UGen) · 2412.06660
(MuMu-LLaMA) · 2310.13289 (SALMONN) · 2607.17079 (SALMONN-2) · 2503.03983 (Audio Flamingo 2)
· 2507.08128 (Audio Flamingo 3) · 2511.10289 (Music Flamingo) · 2503.20215 (Qwen2.5-Omni) ·
2509.17765 (Qwen3-Omni) · 2604.15804 (Qwen3.5-Omni) · 2407.10759 (Qwen2-Audio) · 1306.1461
(GTZAN faults, Sturm 2013) · 2008.07142 (POP909) · 2507.16632 (Step-Audio 2 — source of the
loosely-attributed Gemini MMAU 71.6 figure, verify directly before citing) · 2607.06015
(MusICA-MetaBench) · 2602.04085 (BASS) · 2604.25937 (SongBench) · 2505.10793 (SongEval).

Compiled from two parallel research passes (dataset/performance survey; training-data
survey) plus direct arXiv title verification on a spot-check sample of 8 IDs (all matched,
2026-08-13) and a second spot-check of 4 more IDs (all matched, 2026-08-17). Numbers not
independently re-derived from primary PDFs beyond what's noted above — treat this as a
literature survey, not a replication.
