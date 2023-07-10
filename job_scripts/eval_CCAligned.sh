#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=1:00:00
#SBATCH --job-name=eval_CCAligned
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl

export PATH="$PATH:/home1/s3412768/.local/bin"

# Load modules
# module load Python/3.9.6-GCCcore-11.2.0
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
# module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0 

#load environment
source /home1/s3412768/.envs/nmt2/bin/activate

corpus="CCAligned"

root_dir="/scratch/hb-macocu/NMT_eval/en-sq"
model="Helsinki-NLP/opus-mt-en-sq"
# corpora=("QED" "TED2020" "flores200.devtest" "WikiMatrix")
corpora=("flores200.devtest")
checkpoint="${root_dir}/models/${model}/fine_tuning/${corpus}/checkpoint-*?"

for eval_corpus in ${corpora[@]}; do
    log_file="/scratch/hb-macocu/NMT_eval/en-sq/logs/fine_tune/${corpus}/eval_${eval_corpus}.log"
    python /home1/s3412768/NMT_eval/src/train.py \
        --root_dir $root_dir \
        --checkpoint $checkpoint \
        --train_file $root_dir/data/$corpus.en-sq.tsv.dedup \
        --dev_file $root_dir/data/flores200.dev.en-sq.tsv.dedup \
        --test_file $root_dir/data/$eval_corpus.en-sq.tsv.dedup.test \
        --gradient_accumulation_steps 2 \
        --batch_size 8 \
        --gradient_checkpointing \
        --adafactor \
        --fp16 \
        --exp_type fine_tune \
        --model_name $model \
        --eval \
        --predict 
        &> $log_file 
done