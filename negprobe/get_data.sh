#!/usr/bin/env bash
# Downloads the negated LAMA probes and the LAMA common vocabulary into ./data
set -euo pipefail
mkdir -p data && cd data

# 1) Full negated LAMA release (Kassner & Schuetze 2020, hosted by Facebook)
if [ ! -d negated_data ]; then
  [ -f negated_data.tar.gz ] || curl -fL -o negated_data.tar.gz https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz
  mkdir -p negated_data && tar -xzf negated_data.tar.gz -C negated_data
fi

# 2) Common vocabulary used by LAMA (makes models with different vocabularies comparable)
[ -f common_vocab_cased.txt ] || curl -fL -o common_vocab_cased.txt https://dl.fbaipublicfiles.com/LAMA/common_vocab_cased.txt

# 3) Smaller copy of the probes from Kassner's repo (good for quick pilots)
[ -d LAMA_primed_negated ] || git clone --depth 1 https://github.com/norakassner/LAMA_primed_negated

# 4) LAMA code: contains relation templates and the reference implementation
[ -d LAMA ] || git clone --depth 1 https://github.com/facebookresearch/LAMA

echo "Done. Folders in ./data:"; ls
