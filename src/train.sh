#!/bin/bash
# Job scheduling info, only for us specifically
# SBATCH --time=00:29:59
#SBATCH --job-name=en-sq
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl


# Load modules
module load Python/3.9.6-GCCcore-11.2.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load CUDA/11.7.0

#load environment
source /home1/s3412768/.envs/nmt/bin/activate

root_dir="/scratch/hb-macocu/NMT_eval/en-sq/"
log_file="/scratch/hb-macocu/NMT_eval/en-sq/logs/train.log"

python /home1/s3412768/NMT_eval/src/train.py \
    --root_dir $root_dir \
    --train_file $root_dir/MaCoCuV1.en-sq.tsv.dedup \
    --dev_file $root_dir/flores200.dev.en-sq.tsv.dedup \
    --wandb \
    --model_name Helsinki-NLP/opus-mt-en-sq \
    &> $log_file 
