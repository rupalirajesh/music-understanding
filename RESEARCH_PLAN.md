# Music Understanding in Audio LLMs — Background & Research Plan

A deep-dive study of how audio-language models represent, process, and reason about music —
across all genres — with the goal of building the knowledge base to train a music
understanding model.

Last updated: 2026-07-15

---

# PART 0 — CONCEPTS FROM SCRATCH (onboarding primer)

This part assumes no machine-learning background. It builds every concept the rest of the
plan uses, in order, and ends with a walkthrough of what actually happens when we "run an
experiment." Read this before Part I; Part I then adds the audio-specific detail.

## 0.1 What a language model is

A **large language model (LLM)** is a program trained to do one thing: given a sequence of
text, predict what comes next. It was trained by showing it a huge fraction of the internet
and adjusting its internal numbers (**parameters** or **weights** — billions of them) until
its predictions got good. Everything an LLM appears to "know" is stored implicitly in those
weights.

Some vocabulary that falls out of this:

- **Token**: models don't read letters or words; text is chopped into pieces called tokens
  (roughly ¾ of a word each). "Understanding music" might become three tokens. The model's
  input and output are sequences of tokens.
- **Embedding**: each token is converted into a long list of numbers (a **vector**, e.g.
  4,096 numbers). This vector is the model's internal "meaning" of that token. Similar
  meanings end up as nearby vectors — this is the key trick that makes everything else work.
- **Layers and hidden states**: the model is a stack of identical processing blocks
  (**layers** — typically 30–80 of them). Each layer takes in the vectors from the previous
  layer, lets every position look at every other position (**attention**), and outputs
  refined vectors. The vectors at each intermediate layer are called **hidden states** or
  **activations**. Early layers hold surface-level features; deeper layers hold more
  abstract ones. *We will be reading these hidden states directly — this is what "probing"
  means (§0.5).*
- **Inference**: running a trained model to get answers (as opposed to training it).
- **Temperature**: a knob controlling output randomness. Temperature 0 = always pick the
  most likely next token = deterministic. We run all evals at temperature 0 so results are
  reproducible.
- **Prompt**: the input text we give the model, including instructions and the question.

**How LLMs become assistants:** raw next-token prediction produces a model that continues
text, not one that answers questions. So after pretraining, models get **instruction tuning
/ SFT (supervised fine-tuning)**: further training on examples of (question → good answer)
pairs. This matters enormously for us, because *what's in those pairs determines what the
model can talk about* — a recurring theme in this study (§1.4).

**Open-weights vs closed (API) models:** for some models (Qwen, Audio Flamingo) the weights
are downloadable — we can run them on our own GPU, read their hidden states, and fine-tune
them ("**white-box**"). Others (Gemini, GPT-4o) are only reachable over the internet via an
**API**: we send a prompt + audio, get text back, and see nothing inside ("**black-box**").
Track A works on both; Tracks B and C need open weights.

## 0.2 How audio exists in a computer

- **Waveform / samples**: a microphone measures air pressure thousands of times per second.
  Each measurement is a number; the list of numbers is the waveform. The **sampling rate**
  is how many measurements per second — CD/streaming music uses 44,100 per second (44.1 kHz).
- **Nyquist limit**: a recording sampled at rate R can only represent frequencies up to R/2.
  So 16 kHz audio contains nothing above 8 kHz. Most audio models resample everything to
  16 kHz, silently discarding the top of the spectrum (cymbal shimmer, harmonics, "air").
- **Spectrogram**: instead of pressure-over-time, break the audio into tiny windows
  (~25 ms) and compute which frequencies are present in each — a picture with time on the
  x-axis and frequency on the y-axis. A **mel spectrogram** spaces the frequency axis the
  way human hearing does (finer at low frequencies). This image-like representation is what
  most audio models actually consume, not the raw waveform.
- **Cents**: the unit for fine pitch differences. 100 cents = 1 semitone (adjacent piano
  keys). Trained musicians hear ~5–10 cent differences. When we measure a model's pitch
  discrimination "threshold in cents," we're asking how fine its pitch hearing is.
- **12-TET**: standard Western tuning divides the octave into 12 equal semitones. Much
  real-world music (blues bends, vocal slides, Indian classical) lives *between* these
  steps — one of our probes asks whether models' internal pitch representation is snapped
  to the 12-TET grid.

## 0.3 How a model "hears": the audio LLM (LALM) recipe

An LLM only understands token vectors. To let it hear, you bolt on two components:

```
audio → [encoder] → sequence of vectors → [bridge/adapter] → vectors the LLM can read
                                                                     ↓
                                    LLM reads them like a prefix of "audio words",
                                    then answers your text question about them
```

1. **Audio encoder**: a separate neural network, trained beforehand on audio-only tasks,
   that converts a clip into a sequence of vectors — one vector per small time slice. Think
   of it as the model's ear. Crucially, *the encoder can only pass along what its own
   training taught it to notice*. Most encoders were trained for speech recognition
   (transcribing words), where knowing a note is C vs C# was never necessary. (§1.2 covers
   the encoder families.)
2. **Bridge / adapter / projector**: a small translation layer that maps the encoder's
   vectors into the LLM's embedding space, so the LLM can treat them like tokens.
3. **Token rate**: the encoder emits vectors at some fixed rate, e.g. 12.5 per second =
   one vector per 80 ms. Anything that happens *faster* than that (a quick ornament, a
   grace note) must be summarized into a single vector or it's simply gone. This is a hard
   information bottleneck and one of our central suspects (§1.1, §1.5).

The whole assembly — encoder + bridge + LLM — is called an **audio-language model**
(**LALM** = Large Audio Language Model). "Omni" models are the same idea with audio tokens
integrated more natively.

**The pipeline is a chain of custody for musical information.** The note is in the air →
does it survive resampling to 16 kHz? → does the encoder represent it? → does the bridge
pass it through? → does the LLM *use* it when answering? Our whole methodology (§0.5) is
about finding the link where the chain breaks.

## 0.4 Fine-tuning, LoRA, and why we'll do surgery on models

- **Fine-tuning**: taking a trained model and training it a bit more on your own examples
  to teach it a specific skill. Full fine-tuning updates all billions of weights —
  expensive.
- **LoRA (Low-Rank Adaptation)**: a cheap fine-tuning trick. Freeze the original weights;
  attach small trainable "patch" matrices alongside them and train only those (<1% of the
  parameters). Runs on a single Colab GPU. When the plan says "three-arm LoRA fine-tune"
  (§6), it means: three separate LoRA training runs that patch *different parts* of the
  model (just the LLM side; LLM + encoder; and a control), to see which patch fixes a
  failure — which tells us where the failure lived.
- **Frozen**: a component whose weights are not updated during a given training stage.

## 0.5 The measurement toolkit — the ideas behind L1 / L2 / L3

This is the intellectual core of the study. For any musical property (say, "what key is
this in?"), we ask three separate questions:

**L1 — Is the information physically in the audio?**
Use classical signal processing / small purpose-built tools (**MIR** = Music Information
Retrieval, the pre-LLM field of extracting musical facts from audio: pitch trackers, beat
trackers, key detectors). If a standard key-detection algorithm gets it right, the key is
recoverable from the signal. L1 is our ground-truth sanity floor.

**L2 — Is the information inside the model's head?**
This is **probing**. Take the model's hidden states (§0.1) for our audio clips at some
layer. Train a *tiny, simple* classifier — a **linear probe**, essentially one-step
logistic regression from scikit-learn — to predict the property (the key, the chord
quality) *from those hidden states alone*. If a classifier that simple can read the key off
the model's internal vectors, the information is demonstrably *encoded in there*. We do
this layer by layer ("layer-wise probing") to see where in the stack information appears,
peaks, or fades. The probe must be simple on purpose: a powerful probe could compute the
answer itself, which would prove nothing about the model.

**L3 — Does the model behaviorally use it?**
Just ask the model the question in plain language and score its answer. This is standard
**behavioral evaluation** — no internals needed, works on API models too.

**Why three levels?** Because the *pattern* across them is diagnostic:

- **L1 ✓, L2 ✓, L3 ✗** — the model's ear hears it, but the language side can't talk about
  it. We call this an **alignment gap**. Prediction: cheaply fixable with a little
  fine-tuning (Track C tests exactly this).
- **L1 ✓, L2 ✗** — the signal contains it but the encoder never learned to represent it.
  The problem is the encoder's training or the architecture (token rate, 16 kHz). No amount
  of instruction data fixes this.
- **L2 ✗, L3 ✓** — red flag! The model answers correctly *without* the information being
  in its audio representation — meaning it's guessing from **text priors**: statistical
  knowledge of what answers are plausible ("a question about a sad slow song is probably in
  a minor key"), not listening. The MuChoMusic benchmark famously caught models scoring
  well on music questions *with the audio removed*. This is why every task carries a
  **no-audio control**: run the identical question without the audio; if accuracy barely
  drops, the task was never measuring hearing.

The three experimental tracks map onto this: **Track A** = L3 at scale (behavioral,
black-box, all models). **Track B** = L2 (probing, open models). **Track C** = causal
follow-up — if probing says "the info is in there," prove it by fine-tuning the alignment
and watching behavior recover.

## 0.6 Experimental hygiene concepts (the controls vocabulary)

These terms appear throughout §4 and §9; here's what they mean and why they exist.

- **Synthetic stimuli**: audio we generate ourselves from **MIDI** (a symbolic score format
  — a list of note-on/note-off events, no sound) rendered to audio with a synthesizer
  (fluidsynth + **soundfonts** = libraries of instrument sounds). Because we wrote the
  score, ground truth is *perfect* — we know the exact key, tempo, and every note. It also
  gives **factor control**: we can change one variable (transpose the key) while holding
  everything else constant.
- **MCQ vs open-ended**: multiple-choice questions are easy to score automatically but
  models exploit their structure (position biases, elimination); open-ended answers are
  more natural but need scoring by a text LLM acting as **judge** — which we validate by
  hand-scoring ~100 items and checking the judge agrees.
- **Label/position permutation**: shuffle answer-choice order across items so "always pick
  B" can't score above chance.
- **Paraphrases**: ask each question ≥3 different ways, so we measure the skill, not
  sensitivity to one phrasing.
- **Confusion matrix**: a table of which wrong answers occur. Errors carry signal: calling
  G major "D major" (adjacent on the circle of fifths) is an honest listening error;
  uniformly random errors mean guessing. Same logic for tempo (double/half-tempo "octave
  errors") and technique (bend↔slide vs bend↔palm-mute).
- **Psychometric curve**: borrowed from human hearing science. Vary difficulty continuously
  (pitch difference of 5, 10, 25, 50, 100 cents) and plot accuracy vs difficulty. The point
  where the curve crosses threshold is the model's discrimination limit — far more
  informative than a single accuracy number.
- **Training-data contamination**: famous datasets (GTZAN, MusicCaps) were probably *in*
  these models' training data, so good scores may be memorization. Defense: synthetic
  stimuli and recordings released after the model's training cutoff.
- **Leakage** (in probes): if clips from the same singer appear in both the probe's train
  and test sets, the probe can cheat by recognizing the voice. So we split by singer /
  soundfont / session, never just by clip.
- **Held-out / transfer set**: data never seen during (fine-)tuning, used to check the
  model learned the skill rather than the examples — including transfer from synthetic
  training stimuli to real recordings.

## 0.7 What running an experiment actually looks like

Concretely, the day-to-day loop. Everything is driven by one **manifest** file
(`stimuli.parquet` — a big table): each row = one stimulus, with its audio path, task name,
ground-truth answer, factor values (key, instrument, tempo…), and which controls apply.

**Step 1 — Make stimuli (local machine).** Python scripts generate MIDI, render it through
fluidsynth with several soundfonts, apply manipulations (transposition, detuning), compute
ground truth, and append rows to the manifest. For real-music tasks, rows point into
datasets like MTG-Jamendo with their labels.

**Step 2 — Run models (Track A / L3).** The harness walks the manifest: for each row it
builds the prompt (with paraphrase and choice-permutation variants), sends audio + question
to the model — API call for closed models, local inference on a Colab GPU for open ones —
at temperature 0, and logs *everything* (model version, prompt, audio hash, raw answer) to
a results table. No-audio and wrong-audio control rows run through the same pipe.

**Step 3 — Score & analyze (local).** Parse answers (or run the LLM judge for open-ended),
join against ground truth, and produce per-task accuracy, control-corrected accuracy,
confusion matrices, and psychometric curves. Deliverable: the capability heatmap
(task × model).

**Step 4 — Extract activations (Track B / L2, Colab).** Load an open model, play each
stimulus through it, and save the hidden states at every layer to disk (they're just big
arrays). This is the GPU-hungry step; probing itself is cheap.

**Step 5 — Probe (local).** For each (layer × property): train a linear probe on the saved
activations with scikit-learn, evaluate on the leakage-safe split, plot
probe-accuracy-by-layer. Combine with Step 3 into the L2/L3 dissociation table — the
study's centerpiece.

**Step 6 — Intervene (Track C, Colab).** For the clearest L2-high/L3-low failures:
programmatically generate a small instruction dataset, run the three LoRA arms, re-run the
Step 2–3 evaluation on held-out stimuli, and see which arm closes the gap.

The person running experiments mostly lives in Steps 1–3 and 5 (Python, pandas,
scikit-learn — no deep learning expertise needed) with Steps 4 and 6 being
recipe-following on Colab. The harness in Step 2 is the first thing we build (Phase 0),
and validating it by replicating published benchmark numbers is how we know our plumbing
is trustworthy before we trust any novel result.

## 0.8 Mini-glossary (quick reference)

| Term | One-liner |
|---|---|
| LLM | Text-in/text-out neural network trained to predict the next token |
| Token | Chunk of text (~¾ word); the model's unit of input/output |
| Embedding / vector | List of numbers representing a token's meaning |
| Hidden state / activation | The vectors inside the model at some layer |
| Layer | One of the model's stacked processing blocks (30–80 total) |
| Encoder | Audio-in/vectors-out network — the model's "ear" |
| Bridge / adapter | Translator from encoder vectors to LLM embedding space |
| LALM | Audio LLM: encoder + bridge + LLM |
| Token rate | Audio vectors per second reaching the LLM (e.g. 12.5 Hz = 80 ms each) |
| SFT / instruction tuning | Training on (question → answer) pairs to make a model helpful |
| LoRA | Cheap fine-tuning: train small patch matrices, freeze the rest |
| Linear probe | Tiny classifier reading a property from hidden states → "is it encoded?" |
| L1 / L2 / L3 | Info in signal / in representation / in behavior |
| Alignment gap | L2 ✓ but L3 ✗ — model hears it but can't say it |
| Text prior | Answering from plausibility instead of listening |
| No-audio control | Same question, audio removed — detects text priors |
| MIR | Music Information Retrieval — classical algorithms for musical facts |
| MIDI / soundfont | Symbolic score format / instrument-sound library for rendering it |
| Cents | Fine pitch unit; 100 cents = 1 semitone |
| Psychometric curve | Accuracy vs difficulty; threshold = perceptual limit |
| Contamination | Eval data that was in the model's training set |
| Leakage | Train/test overlap letting a probe cheat (same singer, etc.) |
| Temperature 0 | Deterministic model output (reproducible evals) |
| White-box / black-box | Weights accessible (probe, tune) vs API-only (behavior only) |

---

# PART I — BACKGROUND: HOW THESE MODELS WORK

Read this part top-to-bottom once; it's ordered from signal-level basics to the current SOTA.
Section 1.7 is the reading list in recommended order.

## 1.1 Audio representations 101 (the substrate everything sits on)

- **Waveform**: pressure samples over time. Music is distributed at 44.1 kHz; **almost every
  audio LLM resamples to 16 kHz**, which caps representable frequencies at 8 kHz (Nyquist).
  Fundamentals of musical notes live below ~4 kHz, but upper harmonics, cymbal/air content,
  and production detail are discarded before the model ever "hears" the clip. This is the
  first, easily-forgotten bottleneck.
- **Mel spectrogram**: short-time Fourier transform → perceptually-spaced frequency bins.
  Whisper-family front-ends use 128 mel channels, 25 ms window, 10 ms hop. Mel spacing is
  speech-motivated; adjacent semitones in low registers can fall into the same mel bin.
- **CQT (constant-Q transform)**: log-frequency bins aligned with musical pitch — each
  semitone gets equal resolution. Rarely used in LALM front-ends, but used as a *training
  target* in MERT (below) precisely to inject pitch/harmony inductive bias.
- **Frame rate / token rate** — the single most important number for our study. The encoder
  emits a sequence of vectors at some rate; after pooling/downsampling the LLM sees audio
  "tokens" at:
  - Whisper encoder: 50 Hz native, typically pooled 2–4× → 25 or 12.5 Hz at the LLM
  - Qwen3-Omni's AuT encoder: **12.5 Hz** (80 ms per token)
  - MERT: 75 Hz (it's an encoder for MIR, not an LLM front-end)
  - Neural codecs (EnCodec/DAC): 50–75 Hz × multiple codebooks
  
  For scale: a 16th note at 120 BPM lasts 125 ms ≈ **1–2 tokens** at 12.5 Hz. A vibrato
  cycle (~5–7 Hz) spans a handful of tokens; a fast ornament may fall entirely inside one.
  Anything the encoder doesn't summarize into that one vector is gone. This predicts a
  specific failure class (fast/fine pitch events) that our probes should target.

## 1.2 The four audio-encoder families

Every music-capable LLM is built around one of these. Their pretraining objective determines
what musical information survives into the representation.

**(a) Supervised speech encoders (the dominant choice).**
- **Whisper** (large-v3): encoder-decoder trained on ~5M hours of weakly-supervised
  transcription/translation. The encoder is a superb general audio feature extractor *for
  speech*; its objective never required distinguishing C4 from C#4, only phonemes and
  prosody. Yet it is the backbone of Qwen2-Audio, SALMONN, AF3, and most LALMs.
- **AF-Whisper** (Audio Flamingo 3): Whisper-large-v3 *further pretrained* on sound + music
  + speech (captioning-style objectives) to become a unified encoder; NVIDIA showed one
  unified encoder beats separate speech/sound dual encoders at equal data budget.
- **AuT** (Qwen3-Omni): attention encoder-decoder ASR model trained *from scratch* on
  20M hours of supervised audio; 8× downsampled 128-dim fbank → 12.5 Hz token rate.

*Implication for us:* the field's default encoders are ASR-shaped. Precise-pitch failures
may originate here, not in the LLM.

**(b) Self-supervised (SSL) encoders.**
- **wav2vec 2.0 / HuBERT / BEATs**: masked prediction of quantized/clustered units. General
  audio, speech-leaning.
- **MERT** (the music-specific one, and our main probing substrate): BERT-style masked
  prediction over music audio with **two teachers**: an RVQ-VAE (EnCodec) providing
  discretized *acoustic* targets, and a **CQT reconstruction target** injecting *pitch and
  harmonic* inductive bias. 95M and 330M variants, 24 kHz input, 75 Hz frames. SOTA-ish
  across 14 MIR tasks at release; the standard music representation for linear-probe studies.
- MusicFM, MULE: alternative music SSL encoders worth including in encoder comparisons.

**(c) Contrastive audio-text encoders.**
- **CLAP** (LAION-CLAP, MS-CLAP): dual-tower audio/text embedding trained on caption pairs,
  music variants exist. Gives *clip-level* semantics (good for genre/mood/retrieval), weak
  temporal and pitch detail. Used for zero-shot classification and as retrieval scaffolding.

**(d) Neural audio codecs (discrete tokens).**
- **SoundStream / EnCodec / DAC**: residual vector quantization (RVQ) → parallel streams of
  discrete tokens (~50–75 Hz × 4–32 codebooks) optimized for *reconstruction*, not meaning.
  These power music *generation* (MusicGen; MusicLM pairs semantic tokens from w2v-BERT with
  acoustic codec tokens) and fully-tokenized "omni" models. Codec tokens preserve acoustic
  detail but encode it in a form LLMs find hard to reason over symbolically — the
  semantic-vs-acoustic token tension is a live research thread (e.g., unified tokenizers
  like EntangleCodec, 2026).

## 1.3 The standard LALM recipe (encoder → bridge → LLM)

```
16 kHz waveform → mel/fbank → audio encoder → downsample/pool → bridge → frozen-ish LLM
                                                                  |
                                    (a) linear/MLP projector (LLaVA-style; AF3, Qwen-Audio)
                                    (b) window-level Q-Former (SALMONN)
                                    (c) cross-attention layers (original Flamingo, AF1/AF2)
```

- The **bridge/adapter** maps encoder vectors into the LLM's embedding space; audio arrives
  as a prefix of soft tokens (30 s ≈ 375–750 tokens depending on rate).
- Trend over 2024–2026: away from Q-Former/cross-attention toward simple projectors with
  *better data and curricula* doing the heavy lifting; and toward "omni" models where audio
  tokens share the causal decoder with text natively (Qwen3-Omni's Thinker-Talker MoE:
  Thinker does understanding/text, Talker streams speech out — Talker is irrelevant to
  understanding research).

## 1.4 How they're trained (the part that explains most failures)

Canonical multi-stage curriculum — AF3 is the cleanest fully-open documented example
(five stages):

1. **Encoder pretraining** (before LALM assembly): ASR supervision (Whisper/AuT) or SSL
   (MERT). *Determines what's perceivable at all.*
2. **Alignment pretraining**: freeze encoder + LLM, train only the adapter on
   audio-captioning / ASR / tagging data. Teaches the LLM "what audio tokens mean" at
   caption granularity.
3. **Encoder tuning**: unfreeze encoder (+adapter), broaden to diverse audio tasks.
4. **Full instruction tuning (SFT)**: unfreeze everything; large synthetic QA corpora —
   AF3 used **AudioSkills-XL (~8M QA pairs)**; long-audio stage adds LongAudio-XL (1M+ QA,
   up to 10 min).
5. **Post-training**: chain-of-thought data (AF-Think, 250K), multi-turn chat (AF-Chat), and
   increasingly **RL with verifiable rewards** (GRPO-style). **Music Flamingo** = AF3
   backbone + music-heavy continued training + theory-aware long captions + CoT + RL with
   custom music rewards.

**The critical detail — where the QA data comes from:** instruction pairs are overwhelmingly
*generated by text LLMs from metadata, tags, and captions* of audio datasets — not from
ground-truth musical analysis of the signal. Consequences:
- Supervision is caption-grained ("upbeat jazz with walking bass"), almost never
  note/interval/voicing-grained. Models are never *forced* to resolve precise pitch.
- Text-LLM priors get baked in: the model learns what captions plausibly say about audio
  like this, which is exactly the shortcut MuChoMusic caught models using (answering music
  MCQs decently *without the audio*).
- Skills follow data density: speech tasks ≫ sound events ≫ music semantics ≫ music theory.

## 1.5 Why music specifically is a stress test

1. **16 kHz resampling** throws away the top 1.5 octaves of content before encoding.
2. **ASR-shaped encoders**: objective never rewarded absolute pitch, tuning, voicing.
3. **Token-rate bottleneck**: fast events (ornaments, fast runs, ghost notes, strumming
   detail) compress into 1–2 vectors.
4. **Caption-level supervision**: no note-level ground truth in training.
5. **Text-prior shortcutting**: models answer plausibly without listening (MuChoMusic).
6. **Instruction-following gap on MIR tasks**: CMI-Bench reframed classic MIR tasks
   (key, beat, melody extraction) as instructions and found large gaps vs supervised models.
7. **Long-context degradation**: accuracy drops >10 points beyond ~5 minutes of audio —
   relevant for form/structure tasks.

Each of these is a *hypothesis generator* for Part II: they predict which tasks fail and at
which level (L1/L2/L3) the failure should show up.

## 1.6 State of the art (as of mid-2026)

**Open weights (probeable — our white-box targets):**
- **Music Flamingo** (NVIDIA, Nov 2025): open SOTA for music understanding; AF3 architecture
  (AF-Whisper + projector + Qwen2.5-7B), music-specialized with CoT + RL. Primary target.
- **Audio Flamingo 3** (NeurIPS 2025): fully open (weights + data recipe + code); the best
  *documented* pipeline, which makes it the best model to study and fine-tune.
- **Audio Flamingo Next / AF-Next** (Apr 2026): newest in the series; check weights/recipe
  availability during Phase 0.
- **Qwen3-Omni (30B-A3B MoE)** and **Qwen3.5-Omni** (Apr 2026): strongest general open
  omni-models; AuT encoder @ 12.5 Hz; MoE Thinker. Heavier to probe but important because
  the encoder is *not* Whisper-derived — a natural architecture contrast with AF3.
- **Qwen2-Audio-7B**: older, widely benchmarked; cheap comparison anchor.
- Standalone encoders for L2 work: **MERT-330M**, CLAP, Whisper-enc, AF-Whisper, AuT.

**Closed (behavioral ceilings via API):** Gemini 2.x (native audio), GPT-4o-audio family.
(Claude takes no raw audio — but is useful for the symbolic-music text contrast: ABC/MIDI-as-text.)

**Benchmark state:**
- **MMAU** (ICLR'25) and **MMAU-Pro** (Aug'25, 5.3K expert instances, 49 skills): humans
  77.9%, **no model above ~60%** on Pro. Music is consistently the weakest domain.
- **MuChoMusic** (ISMIR'24): the text-prior exposé; 1.1K validated MCQs.
- **CMI-Bench** (2025): music instruction-following with proper MIR metrics; LALMs far below
  supervised MIR baselines on key/beat/melody.
- **MARBLE**: encoder-level MIR benchmark suite (the L2 reference).
- Recent probing-flavored evals: "core music perception" batteries (2025) and automated
  perception testing on arbitrary music ("Music I Care About," 2026) — closest neighbors to
  our Track A; read before building the harness to avoid duplication and to steal design.

**The headline picture**: models are good at *describing* music (genre, mood,
instrumentation, captions) and bad at *hearing* music precisely (key, intervals, chord
quality, beat structure, technique detail) — i.e., strong caption-space semantics, weak
grounded perception. Our study's job is to turn that folk summary into a mechanistic,
layer-located, causally-tested account.

## 1.7 Reading list (in order)

*Foundations*
1. Whisper paper (Radford et al., 2022) — the default encoder's objective and biases.
2. HuBERT (2021) — masked-prediction SSL; the template MERT follows.
3. **MERT** (ICLR'24) — music SSL with RVQ-VAE acoustic + CQT musical teachers.
4. CLAP (2023) — contrastive audio-text; zero-shot classification.
5. EnCodec (2022) + MusicGen (2023) — codec tokens; the generation-side view.

*Audio LLMs*
6. Qwen2-Audio (2024) — minimal canonical LALM recipe.
7. SALMONN (ICLR'24) — Q-Former bridging, dual encoder.
8. Audio Flamingo 2 (2025) → **Audio Flamingo 3** (NeurIPS'25) — read AF3 closely: 5-stage
   curriculum, AF-Whisper, AudioSkills-XL data generation. This is the recipe you'd adapt
   to train your own model.
9. **Music Flamingo** (2025) — music specialization: data, CoT, RL rewards.
10. Qwen3-Omni technical report (2025) + Qwen3.5-Omni (2026) — AuT encoder, MoE omni design.

*Evaluation & analysis*
11. MuChoMusic (ISMIR'24) — text-prior problem + MCQ methodology.
12. MMAU (ICLR'25), MMAU-Pro (2025) — broad audio reasoning; music subscores.
13. CMI-Bench (2025) — MIR-metric instruction following.
14. MARBLE (2023) — encoder probing benchmark.
15. Surveys: "Towards Holistic Evaluation of LALMs" (2505.15957) and "A Survey of Large
    Audio Language Models: Generalization, Trustworthiness, Outlook" (2605.20266).
16. Recent perception batteries: arXiv 2510.22455 (core music perception in MLLMs),
    2607.06015 (Music I Care About), 2511.05550 (factual music comprehension).

---

# PART II — THE STUDY

## 2. Research questions

**RQ1 (Capability map).** Across *all genres*, which music understanding tasks can current
models perform, at what precision? Low-level perception (notes, intervals, tuning, tempo) →
mid-level structure (key, scale/mode incl. blues/dorian/phrygian, chords, instrument and
vocal technique) → high-level semantics (genre/subgenre, form, production style, mood).

**RQ2 (Mechanism).** Where does musical information live in the pipeline? Is it present in
the encoder representations, and does the LLM *use* it — or answer from text priors and
coarse timbral statistics? How do representations differ across encoder families
(ASR-supervised Whisper/AuT vs music-SSL MERT vs contrastive CLAP)?

**RQ3 (Failure etiology).** For each observed failure, which cause is it?
- **(a) Data coverage** — the skill is learnable by this architecture but undertrained
  (fixable with instruction data);
- **(b) Alignment gap** — the encoder represents it but the LLM can't verbalize it
  (fixable with cheap adapter/LLM-side tuning);
- **(c) Architectural bottleneck** — 16 kHz input, ASR-biased encoder, or token rate
  destroys the information before the LLM sees it (needs encoder/front-end changes).

**RQ4 (Actionable).** What does RQ1–3 imply for training a music understanding model:
which encoder, what token rate, what data mix, which training stage to intervene at?

*(Cultural generalization — e.g., Indian classical, maqam, gamelan — is deferred to an
optional extension module, §8. The machinery built here answers it later for free.)*

### The central logic: a 3-level dissociation

For every property, measure at three levels on the **same stimuli**:

| Level | Measurement | Interpretation when this succeeds but the next fails |
|---|---|---|
| **L1: Signal** | Classical MIR / small supervised model extracts the property | property is physically recoverable from the audio |
| **L2: Representation** | Linear probe on encoder / LLM hidden states decodes it | info is *encoded* but not *verbalized* → alignment gap (RQ3-b) |
| **L3: Behavior** | The LALM answers questions about it | — |

- L2 high / L3 low → alignment gap → cheap fine-tuning fix (confirmed causally in Track C).
- L2 low / L1 high → encoder never learned it → pretraining data or architecture (RQ3-a/c);
  disambiguate with the token-rate and encoder-family analyses (§5.3–5.4).
- L3 high / L2 low → red flag: behavioral success via priors/shortcuts, not listening.

## 3. Task taxonomy — the probe battery

Every experiment is a cell in: **task × stimulus source (synthetic / real) × genre ×
measurement level (L1/L2/L3)**.

### Tier 1 — Low-level perception (synthesizable → perfect ground truth)
1. Pitch: single-note ID (absolute; and relative given a reference), octave placement.
2. Intervals: name the interval; ascending/descending; melodic vs harmonic.
3. **Microtonal discrimination**: higher/lower/same at Δ ∈ {5,10,25,50,100} cents →
   a psychometric curve per model. Also: in-tune vs detuned (quarter-tone flat) note in a
   chord; overall tuning reference (A=440 vs 432).
4. Tempo (BPM estimation, error distribution — octave errors are diagnostic); meter
   (3/4 vs 4/4 vs 6/8 vs 7/8); swing vs straight; syncopation detection.
5. Timbre: instrument ID across families (NSynth) and in-mix (MedleyDB stems).
6. Polyphony: count simultaneous notes; count instruments in a mix.
7. Dynamics & articulation: staccato/legato, crescendo, accent placement.

### Tier 2 — Mid-level structure
8. Key identification (all 24 major/minor), on scales, chord progressions, and real mixes.
9. **Scale/mode identification**: major, natural/harmonic/melodic minor, dorian, phrygian,
   lydian, mixolydian, locrian, blues scale, major/minor pentatonic, whole-tone,
   diminished — presented as (a) bare scales, (b) melodies *in* the mode over a drone,
   (c) full arrangements. (b) and (c) are the diagnostic ones.
10. Chords: quality (maj/min/dim/aug/7ths/extensions), inversions; progressions
    (I–IV–V–I vs I–V–vi–IV vs 12-bar blues vs ii–V–I); modulation detection.
11. **Instrument technique**: guitar — bend, slide, hammer-on/pull-off, palm mute, natural
    harmonics, fingerpicking vs plectrum (GuitarSet, IDMT-SMT-Guitar); strings — pizzicato/
    arco/tremolo/col legno; drums — ghost notes, rimshot, brushes; piano — pedaling.
12. **Vocal technique**: vibrato (and its rate/extent), straight tone, belt, breathy, vocal
    fry, trill, falsetto vs head voice, melisma, growl/scream (metal), rap flow vs sung —
    VocalSet (17 techniques × 20 singers) is purpose-built for the core of this.
13. Continuous-pitch ornament perception across styles: blues bends & blue notes, country
    slides, jazz scoops/falls, gospel runs, pitch-corrected (Auto-Tune) vs natural vocals.
14. Melody extraction/matching: same melody, different arrangement — recognized or not?

### Tier 3 — High-level / semantic (the all-genre layer)
15. Genre & **subgenre** classification: coarse (GTZAN-style 10-way, with caution — known
    label noise) → fine (MTG-Jamendo tags; metal subgenres, EDM subgenres, jazz eras,
    hip-hop regional styles). Include a **genre-cue ablation**: re-render the same piece
    (MIDI → different soundfonts/production) to test whether genre judgments track
    composition or production texture.
16. Era/production style: decade estimation; production techniques — sidechain compression,
    distortion type, reverb size, lo-fi artifacts, sampling vs live drums.
17. Form/structure: verse/chorus/bridge; 12-bar blues cycle; AABA; drop detection (EDM);
    call-and-response; where does the section boundary fall (long-context stress test).
18. Mood/emotion; music captioning quality (vs MusicCaps references); lyric-audio
    integration (does the model use sung lyrics as a shortcut for genre/mood?).
19. Similarity/cover-song: same song, different artist/genre rendition.

## 3.5 Model roster — the experiments per model, and how each is run

This section makes the study concrete: every model we test, which experiments run on it,
and where/how it physically runs. (Tracks are defined in §4–6; task numbers refer to §3.)

### At a glance

| Model | Access | Track A (behavior) | Track B (probing) | Track C (LoRA) | Runs on |
|---|---|---|---|---|---|
| **Music Flamingo** | open weights | full battery | full layer-wise | secondary check | Colab A100, bf16 |
| **Audio Flamingo 3** | open (weights+code+data) | full battery | full layer-wise | **primary platform** | Colab A100, bf16 |
| AF-Next | check in Phase 0 | battery if open | optional | — | TBD |
| Qwen3-Omni-30B-A3B | open weights (MoE) | full battery | stretch goal | — | Colab A100, 8-bit |
| Qwen2.5-Omni-7B | open weights | Tier 1–2 anchor | full layer-wise | — | Colab A100, bf16 |
| Qwen2-Audio-7B | open weights | battery (anchor) | — | — | Colab L4/A100 |
| Gemini 2.x | API only | full battery + long-context | — | — | local harness → API |
| GPT-4o-audio | API only | full battery | — | — | local harness → API |
| Claude | API, no audio in | symbolic contrast only | — | — | local harness → API |
| MERT-330M (+95M) | open encoder | — | full probe suite | — | Colab L4 |
| CLAP (LAION/MS) | open encoder | zero-shot classification | probe suite | — | Colab L4 |
| Whisper-large-v3 encoder | open encoder | — | probe suite | — | Colab L4 |
| AF-Whisper | open encoder | — | probe suite | — | Colab L4 |
| AuT (from Qwen3-Omni) | open, if extractable | — | probe suite | — | Colab A100 |

Every Track A run uses the same manifest, prompts, controls, temperature 0, and logging
(§4 logistics); every Track B run uses the same probe suite and leakage-safe splits (§5);
so all cells of the table are directly comparable.

### White-box LALMs (the models we can open up)

**Music Flamingo — the "best case" open model.**
- *Track A:* the entire §3 battery, Tiers 1–3, all seven controlled contrasts. As the open
  SOTA for music, its failures are the interesting ones: anything Music Flamingo can't do,
  weaker models won't either.
- *Track B:* layer-wise linear probes at three tap points — AF-Whisper encoder layers,
  post-adapter (what the LLM actually receives), and LLM decoder layers at audio positions.
  Probe targets: pitch class, octave, interval, key, mode, chord quality, instrument,
  technique, tempo/meter, genre.
- *Track C:* used as a replication check for whichever arm wins on AF3 (does the same LoRA
  recipe close the same gap on the music-specialized model?).

**Audio Flamingo 3 — the workhorse.**
- *Track A:* full battery. Comparing AF3 vs Music Flamingo isolates what music-heavy
  post-training buys, since they share architecture.
- *Track B:* full layer-wise probing, same tap points as Music Flamingo.
- *Track C:* **all intervention experiments run here first** — it's the only model with
  fully open training code + data recipe. Three arms per selected skill: (i) LLM-side LoRA,
  (ii) LLM + encoder-side adapters, (iii) matched-size unrelated-audio control. Evaluated
  on held-out soundfonts/singers plus a synthetic→real transfer set.

**AF-Next.** Phase 0 action item: check weight/recipe availability. If open, it gets the
Track A battery (is the newest generation better at perception, or just at description?);
Track B only if its architecture is close enough to AF3 to reuse the extraction code.

**Qwen3-Omni-30B-A3B — the architecture contrast.** Its AuT encoder is trained from
scratch (not Whisper-derived) at 12.5 Hz, making it the key test of "is everything
downstream of Whisper's biases?"
- *Track A:* full battery.
- *Track B:* stretch goal only — 30B MoE needs 8-bit quantization and patience; if
  activation extraction proves too painful, Qwen2.5-Omni-7B substitutes (below).

**Qwen2.5-Omni-7B — the practical non-Whisper representative.**
- *Track A:* Tier 1–2 battery (enough to anchor its behavior against its probes).
- *Track B:* full layer-wise probing — this is where the non-Whisper-encoder comparison
  actually happens at manageable cost.

**Qwen2-Audio-7B — the cheap anchor.**
- *Track A only:* full battery. Two purposes: (1) Phase 0 harness validation — replicate
  its published MMAU-music / MuChoMusic numbers to prove our plumbing is right before
  trusting any novel result; (2) a widely-benchmarked reference point connecting our
  numbers to the literature.

### Black-box API models (behavioral ceilings)

**Gemini 2.x (native audio)** and **GPT-4o-audio.**
- *Track A only:* full battery through the same harness (audio + prompt via API,
  temperature 0, everything logged). Gemini additionally gets the long-audio structure
  tasks (task 17) since its context window permits >5 min clips. Caveats: API audio-length
  and format limits get documented per model in Phase 0; no internals means no Track B/C —
  these models give us the field's behavioral ceiling, not mechanism.

**Claude (no raw audio input).**
- *Symbolic contrast arm only* (§4, contrast 6): the same Tier 1–2 questions posed as
  ABC-notation/MIDI-as-text. This measures pure symbolic music knowledge with no listening
  involved — the "knows music theory but can't hear" endpoint that calibrates how much of
  other models' scores could come from theory knowledge alone. The audio-capable models
  get the same symbolic variants, so we can compare each model's audio vs symbolic
  performance head-to-head.

### Standalone encoders (Track B only — the L2 comparison set)

These never answer questions; we probe their representations directly. Same stimuli, same
probe suite, same splits — this is the encoder-family comparison of §5.2, and it's the
cheapest, highest-value part of Track B (all run on a modest Colab GPU in hours).

- **MERT-330M** (and 95M for a size check): the music-SSL reference — expected L2 upper
  bound. Also the substrate for the geometry analyses (§5.3: pitch helix, 12-TET
  quantization) and the 75 Hz side of the temporal-resolution contrast (§5.4).
- **Whisper-large-v3 encoder**: the ASR-shaped baseline — what most LALMs actually hear
  through.
- **AF-Whisper**: Whisper after NVIDIA's audio/music continued pretraining. Probing it vs
  vanilla Whisper isolates exactly what that continued pretraining added.
- **CLAP**: the contrastive clip-level representative — expected strong on genre/mood,
  weak on pitch/temporal targets. Its zero-shot classification mode also serves as a
  simple no-LLM behavioral baseline on Tier 3 tasks.
- **AuT** (if extractable from Qwen3-Omni): the from-scratch ASR encoder at 12.5 Hz — the
  low-frame-rate side of the temporal-resolution contrast against MERT's 75 Hz (hypothesis
  H5's decisive test).

### How each kind of run physically works

- **API models**: the harness runs on the local machine; each manifest row → one API call
  (audio attachment + prompt) → response logged to the results table. Costs money per
  call; no GPU needed.
- **Open 7B-class LALMs**: Colab Pro A100-40GB, bf16, batch 1–4. Track A = generate
  answers; Track B = forward passes with hidden-state hooks, activations saved to Drive as
  fp16, probing done offline with scikit-learn on any machine.
- **Qwen3-Omni-30B MoE**: same, but 8-bit quantized; treat every run as expensive —
  batch the manifest, checkpoint progress to Drive.
- **Standalone encoders**: Colab L4 (or even CPU for CLAP) — trivial compute.
- **Track C LoRA runs**: Colab A100 with `peft`; adapters + eval outputs checkpointed to
  Drive each session; per-skill datasets are 5–20k QA pairs, so single-session training
  runs are feasible.

### Model × phase map

- **Phase 0**: Qwen2-Audio-7B (harness validation); AF3 or Music Flamingo running on
  Colab; AF-Next availability check.
- **Phase 1** (synthetic Tier 1–2, Track A): Music Flamingo, AF3, Qwen3-Omni + Gemini,
  GPT-4o-audio; Claude symbolic arm.
- **Phase 2** (real-music Tier 3 + contrasts): same roster.
- **Phase 3** (Track B): encoder set (MERT, CLAP, Whisper-enc, AF-Whisper, AuT) +
  layer-wise probes on AF3 or Music Flamingo + Qwen2.5-Omni-7B.
- **Phase 4** (Track C): AF3 (all arms), Music Flamingo (replication of winning arm).

## 4. Track A — Behavioral probing (black-box, all models)

**Design principles**
- **Synthesize wherever possible**: MIDI + fluidsynth (multiple soundfonts) for pitch/key/
  mode/chords/tempo with perfect ground truth and orthogonal factor control (pitch ×
  instrument × register × tempo). Microtonal stimuli via pitch-bend or direct synthesis;
  WORLD-vocoder resynthesis to manipulate vocal pitch contours holding timbre constant.
- **Real audio where semantics matter**: genre/production/structure tasks need real mixes —
  MTG-Jamendo, MedleyDB, FMA; freshly-recorded or post-cutoff releases for
  contamination-sensitive claims.
- **MCQ + open-ended, always both**; text-LLM judge for open-ended, validated against ~100
  hand-scored items.
- **Controls on every task** (non-negotiable, per MuChoMusic):
  - *No-audio control*: same question, no audio. If accuracy barely drops, the item measures
    text priors — flag or discard.
  - *Silence / noise / wrong-audio* controls on a subsample.
  - *Choice-order & label permutation*; ≥3 paraphrases per question type.
- **Psychometrics over accuracy**: cents-discrimination thresholds; key confusion matrices
  (circle-of-fifths-adjacent and relative-major/minor errors = real listening; uniform
  errors = guessing); tempo octave-error rates; technique confusions (bend↔slide is an
  honest perceptual error; bend↔palm-mute is not).

**Key controlled contrasts** (each isolates one candidate cause):

| Contrast | Isolates |
|---|---|
| Same progression: piano vs distorted guitar vs synth pad | harmony perception vs timbre |
| Same piece: original mix vs MIDI re-render vs stems-removed | genre = composition vs production texture |
| Mode ID: bare scale vs melody-in-mode vs full arrangement | pitch-set matching vs contextual tonality |
| Transposition ±1–5 semitones | absolute vs relative pitch encoding |
| Natural vs Auto-Tuned vocal, same phrase | continuous-pitch perception |
| Same question: audio vs ABC-notation-as-text | acoustic perception vs symbolic music knowledge |
| Sung lyrics intact vs instrumental version | lyric shortcut in genre/mood answers |

**Logistics.** One manifest-driven eval harness (`stimuli.parquet`: path, task, ground truth,
factors, controls; every answer logged with prompt, seed, temperature=0). Runs locally for
API models, on Colab for open models. 50–200 items/cell; breadth over depth in Phase 1.

## 5. Track B — Representational probing (white-box, open models)

1. **Layer-wise linear probes** on (a) standalone encoders (MERT, CLAP, Whisper-enc,
   AF-Whisper, AuT if extractable) and (b) full LALMs (encoder layers, post-adapter, LLM
   decoder layers at audio positions). Probe targets: pitch class, octave, interval, key,
   mode, chord quality, instrument, technique, tempo/meter, genre. → Probe-accuracy-by-layer
   curves per property.
2. **Encoder-family comparison** (RQ2's sharpest tool): the same probe suite across
   ASR-supervised (Whisper/AuT) vs music-SSL (MERT) vs contrastive (CLAP) encoders
   quantifies what each pretraining objective preserves — directly informs encoder choice
   for your own model.
3. **Geometry & quantization probes**: does pitch form a helix/circle? Is representation
   quantized toward 12-TET (probe error vs distance-from-nearest-semitone — scalloped error
   = 12-TET bias)? Is key encoded absolutely or relatively (train probe transposed, test
   untransposed)?
4. **Temporal-resolution analysis** (RQ3-c): probe for within-frame pitch-contour slope and
   extent; compare properties that live above vs below the token rate (sustained key vs
   fast ornament). If ornament information is unrecoverable at 12.5 Hz but recoverable at
   75 Hz (MERT), that's an architectural answer.
5. **The L2/L3 dissociation table**: probe accuracy vs behavioral accuracy per
   (property × genre) — the quadrant analysis from §2.
6. **Attention/attribution (lightweight)**: do answer tokens attend to the relevant audio
   frames (the ornamented note, the modulation point) or uniformly?

**Colab feasibility:** MERT/CLAP probing is trivial (hours on L4). AF3/Music-Flamingo-7B
activation extraction in bf16 on A100-40GB, batch 1–4, cache activations to Drive as fp16,
probe offline with scikit-learn. Qwen3-Omni MoE-30B needs 8-bit + patience; treat as
stretch goal or use Qwen2.5-Omni-7B as the non-Whisper-encoder representative.

## 6. Track C — Causal intervention (the data-vs-architecture test)

Pick the 3–5 clearest **L2-high/L3-low** failures from Tracks A+B (predicted candidates:
key ID, mode ID, chord quality, technique naming). Then:

1. **Build small instruction sets per skill** (5–20k QA triples each), generated
   programmatically: synthetic MIDI renders for key/mode/chords; VocalSet/GuitarSet labels
   for technique. Zero manual annotation.
2. **Three-arm LoRA fine-tune** on AF3 (best-documented training code):
   - (i) LLM-side LoRA, encoder frozen → tests the alignment-gap hypothesis;
   - (ii) + encoder-side adapters → tests the encoder-gap hypothesis;
   - (iii) control arm tuned on matched-size unrelated audio QA → controls for generic gains.
3. **Evaluate on held-out** instruments/singers/soundfonts (not just held-out clips —
   guard against timbre/artist leakage), plus a *transfer* set (real mixes for a skill
   trained on synthetic stimuli).
4. Interpretation: (i) closes the gap → alignment-only, data-fixable (RQ3-b). Only (ii)
   helps → encoder gap (RQ3-a). Neither, and Track B shows the info missing at all layers →
   architectural (RQ3-c), with the token-rate/encoder-family evidence saying what to change.

This is also a dry run of exactly the fine-tuning stack you'd use to train your own model.

## 7. Datasets & tooling

**Stimuli/data:** synthetic (pretty_midi + fluidsynth, multiple soundfonts; numpy/DDSP for
microtonal; WORLD for vocal resynthesis), NSynth (instruments/pitch), VocalSet (vocal
technique), GuitarSet + IDMT-SMT-Guitar (guitar technique), MedleyDB (stems/melody),
MTG-Jamendo (tags/genre at scale), FMA, GTZAN (legacy comparison only), MusicCaps
(captioning reference), Groove MIDI (drums/microtiming).

**Software:** `mirdata`, `librosa`, `essentia`, `pretty_midi`, `crepe`/`pyin` (L1 pitch),
`madmom` (L1 beat/key), `transformers` + `peft` (models, LoRA), `scikit-learn` (probes),
custom manifest-driven eval harness (first artifact to build — well-suited to building with
Claude Code).

**Compute split:** local/Claude Code = stimulus synthesis, API evals, judging, analysis,
plots. Colab Pro = open-model inference, activation extraction, LoRA training (checkpoint
to Drive; A100 sessions).

## 8. Optional extension module — cultural generalization (deferred)

The same L1/L2/L3 + contrast machinery, applied to traditions with non-12-TET intonation
and different structural grammars: Indian classical (Saraga dataset; tonic/raga/tala/gamaka
tasks), Arabic maqam, gamelan, flamenco. Deferred, not deleted: Tier-1 microtonal
psychometrics (task 3) and the technique/ornament battery (tasks 11–13) already collect the
perceptual prerequisites, so this module later becomes a pure add-on. Full design notes preserved in
`archive/carnatic-module-notes.md`.

## 9. Pitfalls checklist

- [ ] Text-prior shortcutting → no-audio controls on every task.
- [ ] MCQ position/label bias → permute; report bias-corrected accuracy.
- [ ] **Training-data contamination**: GTZAN, MusicCaps, FMA are very likely in these
      models' training data. For high-stakes claims use synthetic stimuli, fresh recordings,
      or post-2026 releases; consider audio fingerprint near-dup checks.
- [ ] Timbre/artist/session leakage in probe train/test splits (split by singer, by
      soundfont, by recording session).
- [ ] Judge-model error on open-ended answers → validate against hand-scored sample.
- [ ] Loudness/duration/quality mismatches across stimulus arms → normalize.
- [ ] Over-reading probes: linear decodability ≠ causal use — that's why Track A (behavior)
      and Track C (intervention) bracket Track B from both sides.
- [ ] Version-pin models; log everything (model rev, prompt, audio hash, sampling params).

## 10. Phased execution

**Phase 0 (wk 1–2): Foundations + infrastructure.**
Work through reading list §1.7 (at least items 3, 8, 9, 11, 13). Build the eval harness +
stimulus manifest schema. Get AF3 or Music Flamingo running on Colab. Replicate 2–3
published MMAU-music / MuChoMusic numbers to validate the harness. Check AF-Next weight
availability.

**Phase 1 (wk 3–6): Synthetic battery — Tier 1–2, Track A.**
Pitch, intervals, cents psychometrics, tempo/meter, key, modes, chords, techniques
(VocalSet/GuitarSet). 2–3 open models + 1–2 API models.
*Deliverable: capability heatmap (task × model) + psychometric curves.*

**Phase 2 (wk 6–9): Real-music & all-genre battery — Tier 3 + contrasts, Track A.**
Genre/subgenre with cue-ablation, production style, structure, the seven controlled
contrasts of §4.
*Deliverable: what "genre understanding" is actually made of; contrast-controlled failure list.*

**Phase 3 (wk 8–12, overlapping): Track B.**
Encoder-family probe comparison; layer-wise probes on one full LALM; 12-TET quantization +
temporal-resolution analyses; the L2/L3 dissociation table.
*Deliverable: "where each skill lives / dies" diagnosis.*

**Phase 4 (wk 12–16): Track C.**
Three-arm LoRA on the top L2-high/L3-low failures.
*Deliverable: causal data-vs-architecture verdict per skill + a validated fine-tuning recipe.*

**Phase 5: Write-up + (optional) cultural-generalization module (§8).**
The dissociation table + intervention results is a paper (ISMIR / NeurIPS D&B / ICASSP);
the harness + instruction sets are releasable artifacts.

## 11. Hypotheses (stated up front)

- **H1**: Description ≫ perception — genre/mood/instrumentation strong; key, chord quality,
  interval naming weak, across all models (predicted by §1.4's caption-grained supervision).
- **H2**: Cents-discrimination thresholds coarse (≥50 cents) for LALMs; MERT probes will
  beat LALM behavior on fine pitch (predicted by ASR encoders + token rate, §1.5).
- **H3**: Genre classification substantially survives no-audio and lyric shortcuts but
  degrades sharply under MIDI re-render → genre skill is largely production-texture +
  prior-driven, not compositional.
- **H4**: Key/mode/chord info decodable from mid encoder layers (L2 high) while behavior
  fails (L3 low) → alignment gap → Track C arm (i) recovers most of it.
- **H5**: Fast-ornament and technique detail unrecoverable even by probes at 12.5 Hz token
  rates but partially recoverable at MERT's 75 Hz → architectural bottleneck class.
- **H6**: Music Flamingo > AF3 > Qwen-Omni on music behavior, but encoder probes (L2) will
  be far more similar than behavior (L3) across models — most variance lives in alignment
  and instruction data, not perception.
