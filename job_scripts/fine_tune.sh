#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=72:00:00
#SBATCH --job-name=fine_tune
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
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

corpus=$1 # corpus to fine-tune on
language=$2 # target language

root_dir="/scratch/hb-macocu/NMT_eval/en-$language"
log_file="/scratch/hb-macocu/NMT_eval/en-$language/logs/fine_tune/train_${corpus}.log"
# if log directory does not exist, create it
if [ ! -d "$root_dir/logs/fine_tune" ]; then
    mkdir -p $root_dir/logs/fine_tune
fi


# for cnr, hr, sr, bs use the same baseline model
if [ $language = 'cnr' ] || [ $language = 'hr' ] || [ $language = 'sr' ] || [ $language = 'bs' ] || [ $language = 'sl' ]; then
    model="Helsinki-NLP/opus-mt-en-sla"
elif [ $language = 'tr' ]; then
    model="Helsinki-NLP/opus-mt-en-trk"
else
    model="Helsinki-NLP/opus-mt-en-${language}"
fi

# for cnr, hr, sr, bs, sl, bg use files ending in .tag
if [ $language = 'hr' ] || [ $language = 'sr' ] || [ $language = 'bs' ] || [ $language = 'sl' ] || [ $language = 'bg' ] || [ $language = 'tr' ]; then
    train_file="$root_dir/data/${corpus}.en-$language.dedup.norm.tsv.tag"
    dev_file="$root_dir/data/flores.dev.en-$language.tsv.tag"
elif [ $language = 'cnr' ]; then 
    train_file="$root_dir/data/${corpus}.en-$language.dedup.norm.srp.tsv.tag"
    dev_file="$root_dir/data/OpusSubs.dev.en-cnr.dedup.norm.srp.tag"
else # is mk mt sq don't need tags
    train_file="$root_dir/data/${corpus}.en-$language.dedup.norm.tsv"
    dev_file="${root_dir}/data/flores.dev.en-${language}.tsv"
fi    

python /home1/s3412768/NMT_eval/src/train.py \
    --root_dir $root_dir \
    --train_file $train_file \
    --dev_file $dev_file \
    --wandb \
    --gradient_accumulation_steps 2 \
    --batch_size 16 \
    --gradient_checkpointing \
    --adafactor \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --learning_rate 1e-5 \
    --exp_type fine_tune\
    --model_name $model \
    --early_stopping 2 \
    --eval_baseline \
    &> $log_file
