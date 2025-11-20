#!/usr/bin/env bash
# Download open-source letter/phoneme audio repos into the audio bucket.
# Targets:
#   - English letters: jona70x/Alphabet-with-sounds
#   - Japanese kana: dar5in/japanese-kana-dataset
#   - Spanish letter/phoneme candidates: rmcpantoja/SUCSpeech

set -euo pipefail

DEST_ROOT="/K3D/K3D_llama_cpp/datasets/audio/phoneme_external"

mkdir -p "$DEST_ROOT"

clone_if_missing() {
  local repo_url="$1"
  local dest="$2"
  if [ -d "$dest/.git" ]; then
    echo "[info] already cloned: $dest"
  else
    git clone "$repo_url" "$dest"
  fi
}

clone_if_missing "https://github.com/jona70x/Alphabet-with-sounds.git" "$DEST_ROOT/Alphabet-with-sounds"
clone_if_missing "https://github.com/dar5in/japanese-kana-dataset.git" "$DEST_ROOT/japanese-kana-dataset"
clone_if_missing "https://github.com/rmcpantoja/SUCSpeech.git" "$DEST_ROOT/SUCSpeech"

echo "[done] Repos cloned under $DEST_ROOT"
