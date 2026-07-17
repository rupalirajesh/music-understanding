#!/usr/bin/env bash
# Download the 3 GM soundfonts (not in git: FluidR3 alone is 141 MB).
# Run from experiments/:  bash scripts/00_download_soundfonts.sh
set -euo pipefail
cd "$(dirname "$0")/../assets/soundfonts"

if [ ! -f "MuseScore_General.sf3" ]; then
  curl -L -o MuseScore_General.sf3 \
    "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3"
fi

if [ ! -f "TimGM6mb.sf2" ]; then
  curl -L -o TimGM6mb.sf2 \
    "https://raw.githubusercontent.com/craffel/pretty-midi/main/pretty_midi/TimGM6mb.sf2"
fi

if [ ! -f "FluidR3 GM2-2.SF2" ]; then
  curl -L -o fluid-soundfont.zip \
    "https://ftp.osuosl.org/pub/musescore/soundfont/fluid-soundfont.zip"
  unzip -o fluid-soundfont.zip "FluidR3 GM2-2.SF2"
  rm -f fluid-soundfont.zip
fi

ls -lh
echo "OK — 3 soundfonts in place"
