#!/usr/bin/env bash
# Downloads the Indian Liver Patient Dataset (ILPD) from the UCI ML Repository (CC BY 4.0).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw
curl -sSL -o data/raw/ilpd.zip \
  "https://archive.ics.uci.edu/static/public/225/ilpd+indian+liver+patient+dataset.zip"
unzip -o -q data/raw/ilpd.zip -d data/raw
rm data/raw/ilpd.zip
echo "Saved: data/raw/Indian Liver Patient Dataset (ILPD).csv"
