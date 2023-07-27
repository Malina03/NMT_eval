#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=1:30:00
#SBATCH --job-name=pred_hbs
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G
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

train_corpus="MaCoCuV2"

root="/scratch/hb-macocu/NMT_eval"
root_dir="${root}/en-hbs"
checkpoint=$root_dir/models/fine_tune/$train_corpus/checkpoint-*
model="Helsinki-NLP/opus-mt-en-sla"

# languages=("bg" "bs" "cnr" "hr"	"is" "mk" "mt" "sl" "sq" "sr" "tr")

languages=("bs" "cnr" "hr" "sr")


for language in "${languages[@]}"; do

    if [ $language = 'cnr' ]; then
        test_corpus="OpusSubs"
    else
        test_corpus="flores_devtest"
    fi

    log_file="${root_dir}/logs/eval/${train_corpus}/${language}/eval_${test_corpus}.log"
    # if log directory does not exist, create it
    if [ ! -d "${root_dir}/logs/eval/${train_corpus}/${language}" ]; then
        mkdir -p $root_dir/logs/eval/$train_corpus/$language
    fi
    
    if [ $language = 'cnr' ]; then 
        test_file="${root}/en-cnr/data/OpusSubs.test.en-cnr.dedup.norm.tsv.srp.tag"
    else
        test_file="${root}/en-${language}/data/${test_corpus}.en-${language}.tsv.tag"
    fi  
    
    python /home1/s3412768/NMT_eval/src/train.py \
        --root_dir $root_dir \
        --train_file $test_file \
        --dev_file $test_file \
        --test_file $test_file\
        --gradient_accumulation_steps 2 \
        --batch_size 16 \
        --gradient_checkpointing \
        --adafactor \
        --exp_type "eval/${train_corpus}/${language}" \
        --checkpoint $checkpoint \
        --model_name $model \
        --eval \
        --predict \
        &> $log_file 
    

done
