"""PROJECT_STATE.md next action 23 -- the pitch/interval battery on REAL
recordings (not synthesized MIDI), sourced from MedleyDB's melody
annotations. Every task in this project up to now used stimuli synthesized
from pretty_midi + fluidsynth specifically because that gives free, perfect
ground truth (RESEARCH_PLAN.md Sec3, "Tier 1 -- synthesizable -> perfect
ground truth") -- this module trades that for real recorded audio (real
instruments/vocals, real mixes, real production) against MedleyDB's
expert-annotated melody F0 curve as ground truth instead.

Applies the SAME already-trained front-ends to the resulting snippets,
unmodified -- both are audio-derived-only functions with no dependency on
this project's synthetic manifest, so they work on any wav path:
  scripts/render_f0_contours.render_zoom   (Track D-zoom's image)
  musicprobe/f0_text.f0_segments/text_for_segments  (Track E's text)
The point isn't a new front-end, it's the same fix tested on audio those
front-ends were never rendered against during training -- does the D-zoom/E
trick generalize past clean synthetic tones to real recordings?

------------------------------------------------------------------------------
BLOCKED on data access (2026-08-12): MedleyDB's audio requires a manual
Zenodo access request (https://zenodo.org/record/2628782, "request access"
button) -- a human-approval step, not something this session can complete.
Rupali is requesting access directly; this module is written and verified
against a SYNTHETIC STAND-IN (see musicprobe/tests informal check, run via
`python -m musicprobe.real_music_medleydb --selftest`), not real MedleyDB
audio, which doesn't exist on this machine yet. Once
`data_home` points at a real extracted MedleyDB-Melody folder:
  python -m musicprobe.real_music_medleydb --data-home /path/to/MedleyDB-Melody

NO KEY GROUND TRUTH: mirdata's medleydb_melody Track exposes artist/title/
genre/melody1-3, but no key label -- key_id-on-real-music stays unresolved.
Both GiantSteps' original Beatport CDN and its HuggingFace re-upload
(m-a-p/GS) were tested directly 2026-08-12 and are dead/audio-less. If a
working key-labeled real-audio source turns up later, wire it in separately
-- this module deliberately does not attempt to fake key ground truth from
metadata (e.g. guessing from track titles) because that would be exactly the
kind of unverified ground truth this project's whole L1/L2/L3 discipline
exists to avoid.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .config import EXP_ROOT, MANIFEST_DIR
from .theory import NOTE_NAMES, INTERVALS
from .l1_baselines import f0_to_midi
from .prompts import build_prompt

OUT_AUDIO_DIR = Path("stimuli") / "real_music_medleydb"   # relative to EXP_ROOT, same
                                                            # convention as every synth task
MANIFEST_PATH = MANIFEST_DIR / "real_music_medleydb.parquet"
JOBS_PATH = MANIFEST_DIR / "real_music_medleydb_jobs.parquet"

MIN_SEG_DUR = 0.35     # seconds -- shorter than this isn't a "clear" sustained note
CENTS_TOL = 35.0       # a segment stays "the same note" while pitch drifts <35 cents
                        # (real melodies have vibrato/portamento synthetic tones don't)
PAD = 0.08              # seconds of padding before/after a snippet, avoids clipping onsets
MAX_INTERVAL_GAP = 1.5  # seconds -- adjacent segments further apart than this aren't a
                        # clean "two notes played" interval question


def note_segments(times: np.ndarray, freqs: np.ndarray) -> list[dict]:
    """Group a continuous F0 curve into sustained-note segments -- same
    "grouped voiced run" idea as musicprobe.f0_text.f0_segments, but tracks
    onset/offset TIME (not just a median Hz) since these become audio-clip
    boundaries here, and requires each frame to stay within CENTS_TOL of the
    segment's running median (a real melody's F0 wanders continuously; a
    plain voiced/unvoiced split, f0_text's criterion, would merge an entire
    melodic phrase into one "segment")."""
    segs = []
    cur_t, cur_f = [], []

    def _flush():
        if not cur_t:
            return
        dur = cur_t[-1] - cur_t[0]
        if dur >= MIN_SEG_DUR:
            med = float(np.median(cur_f))
            segs.append({"start": cur_t[0], "end": cur_t[-1], "hz": med,
                        "midi": f0_to_midi(med)})

    for t, f in zip(times, freqs):
        if f <= 0:
            _flush(); cur_t, cur_f = [], []
            continue
        if cur_f:
            med = np.median(cur_f)
            cents = 1200.0 * np.log2(f / med)
            if abs(cents) > CENTS_TOL:
                _flush(); cur_t, cur_f = [], []
        cur_t.append(t); cur_f.append(f)
    _flush()
    return segs


def _interval_label(semi: int) -> tuple[int, str] | None:
    """Fold to the 1..12 range INTERVALS covers, same rule as
    l1_baselines.interval_estimate, so ground_truth vocabulary matches the
    synthetic interval_id task exactly."""
    if semi == 0:
        return None   # unison -- not a target label in this project's INTERVALS map
    folded = ((abs(semi) - 1) % 12) + 1
    if folded not in INTERVALS:
        return None
    return folded, INTERVALS[folded][1]


def _write_snippet(y: np.ndarray, sr: int, t0: float, t1: float, out_path: Path) -> bool:
    i0, i1 = max(0, int((t0 - PAD) * sr)), min(len(y), int((t1 + PAD) * sr))
    if i1 - i0 < int(0.2 * sr):
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y[i0:i1], sr)
    return True


def build_manifest(data_home: str, n_pitch=60, n_interval=60, seed=0,
                   exp_root: Path = EXP_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    import mirdata
    ds = mirdata.initialize("medleydb_melody", data_home=data_home)
    rng = np.random.default_rng(seed)

    pitch_rows, interval_rows = [], []
    for track_id in ds.track_ids:
        tr = ds.track(track_id)
        mel = tr.melody1
        if mel is None:
            continue
        segs = note_segments(mel.times, mel.frequencies)
        y, sr = None, None
        for i, s in enumerate(segs):
            pitch_rows.append({"track_id": track_id, "title": tr.title, "artist": tr.artist,
                              "genre": tr.genre, "seg_idx": i, **s})
        for i in range(len(segs) - 1):
            a, b = segs[i], segs[i + 1]
            if b["start"] - a["end"] > MAX_INTERVAL_GAP:
                continue
            semi = round(b["midi"] - a["midi"])
            lab = _interval_label(semi)
            if lab is None:
                continue
            interval_rows.append({"track_id": track_id, "title": tr.title, "artist": tr.artist,
                                 "genre": tr.genre, "seg_idx": i, "start": a["start"],
                                 "end": b["end"], "semitones": semi,
                                 "interval_short": INTERVALS[lab[0]][0], "interval": lab[1]})

    pitch_pool = pd.DataFrame(pitch_rows)
    interval_pool = pd.DataFrame(interval_rows)
    print(f"[real_music] candidate pool: {len(pitch_pool)} note segments across "
          f"{pitch_pool.track_id.nunique() if len(pitch_pool) else 0} tracks, "
          f"{len(interval_pool)} adjacent-pair intervals")

    pitch_sample = pitch_pool.sample(min(n_pitch, len(pitch_pool)), random_state=seed) \
        if len(pitch_pool) else pitch_pool
    interval_sample = interval_pool.sample(min(n_interval, len(interval_pool)), random_state=seed) \
        if len(interval_pool) else interval_pool

    man_rows, job_rows = [], []
    track_audio_cache: dict[str, tuple] = {}

    def _audio(track_id):
        if track_id not in track_audio_cache:
            track_audio_cache[track_id] = ds.track(track_id).audio
        return track_audio_cache[track_id]

    for r in pitch_sample.itertuples():
        y, sr = _audio(r.track_id)
        stim_id = f"real_music/pitch/{r.track_id}_{r.seg_idx}"
        rel_path = str(OUT_AUDIO_DIR / f"pitch_{r.track_id}_{r.seg_idx}.wav")
        if not _write_snippet(y, sr, r.start, r.end, exp_root / rel_path):
            continue
        note = NOTE_NAMES[int(round(r.midi)) % 12]
        man_rows.append({"stimulus_id": stim_id, "task": "pitch_note_id",
                         "audio_path": rel_path, "ground_truth": note,
                         "source": "medleydb_melody1", "track_id": r.track_id,
                         "title": r.title, "artist": r.artist, "genre": r.genre})

    for r in interval_sample.itertuples():
        y, sr = _audio(r.track_id)
        stim_id = f"real_music/interval/{r.track_id}_{r.seg_idx}"
        rel_path = str(OUT_AUDIO_DIR / f"interval_{r.track_id}_{r.seg_idx}.wav")
        if not _write_snippet(y, sr, r.start, r.end, exp_root / rel_path):
            continue
        man_rows.append({"stimulus_id": stim_id, "task": "interval_id",
                         "audio_path": rel_path, "ground_truth": r.interval,
                         "source": "medleydb_melody1", "track_id": r.track_id,
                         "title": r.title, "artist": r.artist, "genre": r.genre})

    manifest = pd.DataFrame(man_rows)
    # wrong_audio control (Rupali's call, 2026-08-12): reuse this project's
    # existing contamination check rather than inventing a new one -- pair
    # every row with a mismatched real clip from a DIFFERENT track, asked
    # the SAME question, scored against the ORIGINAL ground truth. If a
    # model still confidently answers a famous-song question right on the
    # wrong audio, that's text-prior recall, not listening.
    for i, r in enumerate(manifest.itertuples()):
        others = manifest[(manifest.task == r.task) & (manifest.stimulus_id != r.stimulus_id)]
        if len(others) == 0:
            continue
        wrong = others.sample(1, random_state=seed + i).iloc[0]
        for pi in range(3):   # 3 paraphrases, matching this project's TEMPLATES convention
            prompt = build_prompt(r.task, pi, "open", None)
            job_rows.append({"job_id": f"{r.stimulus_id}::audio::open::p{pi}",
                            "stimulus_id": r.stimulus_id, "task": r.task,
                            "condition": "audio", "format": "open", "paraphrase_idx": pi,
                            "prompt": prompt, "ground_truth": r.ground_truth,
                            "audio_path": r.audio_path})
            job_rows.append({"job_id": f"{r.stimulus_id}::wrong_audio::open::p{pi}",
                            "stimulus_id": r.stimulus_id, "task": r.task,
                            "condition": "wrong_audio", "format": "open", "paraphrase_idx": pi,
                            "prompt": prompt, "ground_truth": r.ground_truth,
                            "audio_path": wrong.audio_path})
    jobs = pd.DataFrame(job_rows)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(MANIFEST_PATH, index=False)
    jobs.to_parquet(JOBS_PATH, index=False)
    print(f"[real_music] wrote {len(manifest)} stimuli -> {MANIFEST_PATH}")
    print(f"[real_music] wrote {len(jobs)} jobs ({jobs.condition.value_counts().to_dict()}) "
          f"-> {JOBS_PATH}")
    return manifest, jobs


def _selftest():
    """Synthetic stand-in for real MedleyDB audio (not available on this
    laptop -- see module docstring). Builds a fake 3-note melody (a major
    third then a perfect fifth, both well inside MAX_INTERVAL_GAP) with
    vibrato, confirms note_segments recovers 3 clean segments and the
    correct two interval labels, and that render_zoom/f0_text run against
    the resulting audio without crashing -- exactly the checks that matter
    before trusting this against the real dataset."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from render_f0_contours import render_zoom          # noqa: E402 -- Track D-zoom, generic
    from musicprobe.f0_text import f0_segments, text_for_segments  # noqa: E402 -- Track E, generic

    sr = 16000
    notes_hz = [261.63, 329.63, 392.00]   # C4, E4 (+4 semi, major third), G4 (+3 semi, minor third)
    dur = 0.6
    y = []
    t_cursor = 0.0
    times, freqs = [], []
    for hz in notes_hz:
        n = int(dur * sr)
        t = np.arange(n) / sr
        vibrato = 1 + 0.003 * np.sin(2 * np.pi * 5 * t)   # mild vibrato, stays inside CENTS_TOL
        y.append(0.3 * np.sin(2 * np.pi * hz * vibrato * t))
        times.extend((t_cursor + t).tolist())
        freqs.extend((hz * vibrato).tolist())
        t_cursor += dur
    y = np.concatenate(y).astype(np.float32)
    times, freqs = np.array(times), np.array(freqs)

    segs = note_segments(times, freqs)
    assert len(segs) == 3, f"expected 3 segments, got {len(segs)}: {segs}"
    m1 = round(segs[1]["midi"] - segs[0]["midi"])
    m2 = round(segs[2]["midi"] - segs[1]["midi"])
    assert m1 == 4, f"expected +4 semitones (C4->E4, major third), got {m1}"
    assert m2 == 3, f"expected +3 semitones (E4->G4, minor third), got {m2}"
    lab1 = _interval_label(m1); lab2 = _interval_label(m2)
    assert lab1[1] == "major third", lab1
    assert lab2[1] == "minor third", lab2
    print("[real_music selftest] note_segments + interval labeling: OK")

    import tempfile
    tmp = Path(tempfile.mkdtemp())
    wav_path = tmp / "fake_melody.wav"
    sf.write(wav_path, y, sr)
    render_zoom(wav_path, tmp / "zoom.png", "cents_discrimination")
    assert (tmp / "zoom.png").exists()
    txt = text_for_segments(f0_segments(wav_path))
    assert "Hz" in txt or "no clear" in txt
    print("[real_music selftest] render_zoom (Track D-zoom) + f0_text (Track E): OK, ran "
          "unmodified against a non-synthetic-battery wav file")
    print("[real_music selftest] ALL CHECKS PASSED (synthetic stand-in -- real MedleyDB "
          "audio not available on this machine, see module docstring)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-home", default=None)
    ap.add_argument("--n-pitch", type=int, default=60)
    ap.add_argument("--n-interval", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        if a.data_home is None:
            raise SystemExit("--data-home required (path to extracted MedleyDB-Melody folder) "
                            "-- or run --selftest to verify the pipeline without real data.")
        build_manifest(a.data_home, a.n_pitch, a.n_interval, a.seed)
