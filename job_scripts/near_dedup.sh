#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl
set -eu -o pipefail

# Script for applying bifixer and bicleaner.

export PATH="$PATH:/home1/s3412768/.local/bin"
export DATA_DIR="/scratch/hb-macocu/NMT_eval/en-sq/data"
#load environment
source /home1/s3412768/.envs/nmt/bin/activate

# Load modules
module load Python/3.9.6-GCCcore-11.2.0

python $HOME/NMT_eval/src/multiple_near_dedup.py \
    --main_files $DATA_DIR/MaCoCuV1.en-sq.tsv $DATA_DIR/CCAligned.en-sq.tsv \
    --other_files $DATA_DIR/TED2020.en-sq.tsv $DATA_DIR/QED.en-sq.tsv $DATA_DIR/WikiMatrix.en-sq.tsv $DATA_DIR/flores200.devtest.en-sq.tsv $DATA_DIR/flores200.dev.en-sq.tsv \
    --dedup either \