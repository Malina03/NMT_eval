#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=12:00:00
#SBATCH --job-name=MacoCuV1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl

export PATH="$PATH:/home1/s3412768/.local/bin"

# Load modules
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0


export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0 

#load environment
source /home1/s3412768/.envs/nmt2/bin/activate

corpus="MaCoCuV1"

root_dir="/scratch/hb-macocu/NMT_eval/en-sq"
# log_file="/scratch/hb-macocu/NMT_eval/en-sq/logs/fine_tune/train_${corpus}.log"

python /home1/s3412768/NMT_eval/src/train.py \
    --root_dir $root_dir \
    --train_file "$root_dir/data/${corpus}.en-sq.tsv.dedup" \
    --dev_file $root_dir/data/flores200.dev.en-sq.tsv.dedup \
    --wandb \
    --gradient_accumulation_steps 2 \
    --batch_size 16 \
    --gradient_checkpointing \
    --adafactor \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --learning_rate 1e-5 \
    --exp_type ft2 \
    --model_name Helsinki-NLP/opus-mt-en-sq \
    --early_stopping 2 \
    --eval_baseline \