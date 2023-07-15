#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=2:00:00
#SBATCH --job-name=test-bs-v2
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
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

corpus="MaCoCuV2"
language="bs"
model="Helsinki-NLP/opus-mt-tc-base-en-sh"


root_dir="/scratch/hb-macocu/NMT_eval/en-${language}"
log_file="/scratch/hb-macocu/NMT_eval/en-${language}/logs/fine_tune/train_${corpus}.log"
# if log directory does not exist, create it
if [ ! -d "$root_dir/logs/fine_tune" ]; then
    mkdir -p $root_dir/logs/fine_tune
fi

python /home1/s3412768/NMT_eval/src/train.py \
    --root_dir $root_dir \
    --train_file $root_dir/data/$corpus.en-$language.cyrillic.dedup.norm.tsv.tag \
    --train_file_2 $root_dir/data/$corpus.en-$language.latin.dedup.norm.tsv.tag \
    --dev_file $root_dir/data/flores_dev.en-$language.tsv.tag \
    --wandb \
    --gradient_accumulation_steps 2 \
    --batch_size 16 \
    --gradient_checkpointing \
    --adafactor \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --learning_rate 1e-5 \
    --exp_type fine_tune \
    --model_name $model \
    --early_stopping 2 \
    --eval_baseline \
    &> $log_file
