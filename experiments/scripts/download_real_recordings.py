"""Fetch the real-recordings battery (PROJECT_STATE.md next action 23's
interim set) from Wikimedia Commons, driven by
manifests/real_recordings_manifest.csv's source_url column -- audio isn't
committed to git (experiments/stimuli/ is gitignored, same policy as the
synthetic battery: regenerate, don't commit binary audio), so this script IS
the "where the data lives" step for this set, the way pretty_midi+fluidsynth
synthesis is for everything else.

  python scripts/download_real_recordings.py
  python scripts/download_real_recordings.py --force   # re-download even if present

All 7 URLs were tested live 2026-08-12 (see PROJECT_STATE next action 23).
One file (the Chopin Nocturne) is stored as .wav in the manifest, not .ogg --
its original Commons upload is a skeleton-multiplexed Ogg Vorbis stream that
libsndfile can't parse directly; this script re-encodes it via librosa's
audioread fallback the same way it was fixed originally, so a fresh run
reproduces the same fix instead of silently leaving a broken .ogg in place.
"""
import argparse
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from musicprobe.config import EXP_ROOT, MANIFEST_DIR  # noqa: E402

MANIFEST_PATH = MANIFEST_DIR / "real_recordings_manifest.csv"
UA = "Mozilla/5.0 (research download; see PROJECT_STATE.md next action 23)"


def _commons_file_url(page_url: str) -> str:
    """page_url is a commons.wikimedia.org/wiki/File:... page; resolve it to
    the actual upload.wikimedia.org direct file URL via the MediaWiki API."""
    title = "File:" + page_url.rsplit("File:", 1)[1]
    api = ("https://commons.wikimedia.org/w/api.php?action=query&titles="
          f"{urllib.parse.quote(title)}&prop=imageinfo&iiprop=url&format=json")
    req = urllib.request.Request(api, headers={"User-Agent": UA})
    import json
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    page = next(iter(data["query"]["pages"].values()))
    return page["imageinfo"][0]["url"]


def download(force: bool = False):
    man = pd.read_csv(MANIFEST_PATH)
    for r in man.itertuples():
        out_path = EXP_ROOT / r.audio_path
        # the Chopin row's audio_path is already .wav (re-encoded); download
        # its source .ogg to a temp name first, then convert.
        needs_reencode = out_path.suffix == ".wav"
        fetch_path = out_path.with_suffix(".ogg") if needs_reencode else out_path
        if fetch_path.exists() and not force and not needs_reencode:
            print(f"  {out_path.name}: already present, skipping (--force to redo)")
            continue
        if out_path.exists() and not force:
            print(f"  {out_path.name}: already present, skipping (--force to redo)")
            continue
        url = _commons_file_url(r.source_url)
        time.sleep(2)   # Commons' upload CDN 429'd mid-run during testing 2026-08-12 --
                         # a couple seconds between requests avoided it on retry
        print(f"  {out_path.name}: fetching {url}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        fetch_path.write_bytes(data)
        if needs_reencode:
            import librosa
            import soundfile as sf
            y, sr = librosa.load(fetch_path, sr=None, mono=False)
            sf.write(out_path, y.T if y.ndim > 1 else y, sr)
            fetch_path.unlink()
        print(f"    -> {out_path} ({out_path.stat().st_size} bytes)")
    print(f"[download_real_recordings] done, {len(man)} files per manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    download(a.force)
