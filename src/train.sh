#!/bin/bash
# Job scheduling info, only for us specifically
# SBATCH --time=00:29:59
#SBATCH --job-name=en-sq
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl

export PATH="$PATH:/home1/s3412768/.local/bin"
#load environment
source /home1/s3412768/.envs/nmt/bin/activate

# Load modules
module load Python/3.9.6-GCCcore-11.2.0

root_dir="/scratch/hb-macocu/NMT_eval/en-sq/"
log_file="/scratch/hb-macocu/NMT_eval/en-sq/logs/train.log"

python /home1/s3412768/NMT_eval/src/train.sh \
    --root_dir $root_dir \
    --train_file $root_dir/MaCoCuV1.en-sq.tsv.dedup \
    --dev_file $root_dir/flores200.dev.en-sq.tsv.dedup \
    --wandb \
    --model_name Helsinki-NLP/opus-mt-en-sq \
    &> $log_file 
