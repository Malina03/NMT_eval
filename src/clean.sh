#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=00:09:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rikvannoord@gmail.com
set -eu -o pipefail

# Script for applying bifixer and bicleaner.

# Load CUDA module to be sure
module load CUDA

export FILE=$1 # Input file with tab-separated sentences
export SRC=$2 # Iso code for source language
export TRG=$3 # Iso code for target language

echo "Processing $1"

# Maybe you have to download the models first
# See explanation: https://github.com/bitextor/bicleaner-ai#parameters
export MODELS="bicleaner_models"

JOBS=16
# Set bicleaner jobs to number of GPUs if GPUs are available
BICLEANER_JOBS=1

score (){
    export CUDA_VISIBLE_DEVICES=$1
    export CUDA_VISIBLE_DEVICES=$((CUDA_VISIBLE_DEVICES-1))

    bicleaner-ai-classify \
        --scol 1 --tcol 2 \
        --disable_minimal_length \
        --quiet \
        - - $MODELS/$SRC-$TRG/metadata.yaml
}

export -f score

# 1. Apply bifixer
# don't do resegmentation
# 2. Score with bicleaner
# don't discard by minimal length (mostly recommended for web-crawled)
# 3. Filter by bicleaner score
# 4. Sort by bifixer hash and bifixer score
# 5. Uniq by hash to near-deduplicate
cat $FILE | parallel -k --pipe -j $JOBS --block 10M \
    bifixer --quiet \
    --scol 1 --tcol 2 \
    --ignore_segmentation \
    --aggressive_dedup \
    $FILE - $SRC $TRG \
| parallel --pipe -k -j$BICLEANER_JOBS score {%} > ${FILE}.clean || true

# columns in .clean: sent1,sent2,bifixer-hash,bifixer-score,bicleaner-score

# Sleep 5 seconds to maybe avoid an error
sleep 5 || true

echo "Now filter the data"

# Only keep sentences with a bicleaner score of >= 0.5 and save to separate file
cat ${FILE}.clean | awk -F"\t" '$5 >= 0.5' | cut -f1,2 > ${FILE}.clean.filter



