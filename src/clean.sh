#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl
set -eu -o pipefail

# Script for applying bifixer and bicleaner.

export PATH="$PATH:/home1/s3412768/.local/bin"
#load environment
source /home1/s3412768/.envs/nmt/bin/activate

# Load modules
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA
module load CMake/3.22.1-GCCcore-11.2.0
module load hunspell/1.7.1-GCCcore-11.2.0
module load GCC/11.2.0
module load GCCcore/11.2.0

# export FILE=$1 # Input file with tab-separated sentences
# export SRC=$2 # Iso code for source language
# export TRG=$3 # Iso code for target language

FILE="/scratch/hb-macocu/NMT_eval/TED2020.en-sq.tsv.dedup"
SRC="en"
TRG="sq"

# Maybe you have to download the models first
# See explanation: https://github.com/bitextor/bicleaner-ai#parameters
MODELS="/home1/s3412768/NMT_eval/bicleaner_models"

JOBS=16
# Set bicleaner jobs to number of GPUs if GPUs are available
BICLEANER_JOBS=1

score (){
    CUDA_VISIBLE_DEVICES=$1
    CUDA_VISIBLE_DEVICES=$((CUDA_VISIBLE_DEVICES-1))

    bicleaner-ai-classify \
        --scol 1 --tcol 2 \
        --disable_minimal_length \
        --quiet \
        --$MODELS/$SRC-$TRG/metadata.yaml
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


# Only keep sentences with a bicleaner score of >= 0.5 and save to separate file
cat ${FILE}.clean | awk -F"\t" '$5 >= 0.5' | cut -f1,2 > ${FILE}.clean.filter



