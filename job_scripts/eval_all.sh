#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=02:00:00
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G

module purge
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
source $HOME/.envs/nmt_eval/bin/activate

set -eu -o pipefail

# languages=("bg" "bs" "cnr" "hr"	"is" "mk" "mt" "sl" "sq" "sr" "tr")
# languages=("bs" "cnr" "hr" "is" "sr" "tr")

# train_corpus="MaCoCuV2"
train_corpus="MaCoCuV1"
languages=("hr" "tr")

# # Calculate all metrics between two files
# out=$1 # File produced by model
# eval=$2 # File from which to extract ref and src
# lang=$3 # Language of the target file (needed for BERT-score)

root="/scratch/hb-macocu/NMT_eval"

for lang in "${languages[@]}"; do

    
    if [ $lang = 'cnr' ]; then
        eval=$root/en-$lang/data/OpusSubs.test.en-cnr.dedup.norm.tsv
        out=$root/en-$lang/logs/eval/$train_corpus/OpusSubs_predictions.txt
    else
        eval=$root/en-$lang/data/flores_devtest.en-$lang.tsv
        out=$root/en-$lang/logs/eval/$train_corpus/flores_devtest_predictions.txt

    fi

    ref=$eval.ref
    src=$eval.src

    # check if ref and src files exist and create them if not
    if [[ ! -f $ref ]]; then
        echo "Reference file $ref not found, create it"
        # First check if the file exists in the data folder
        if [[ -f $eval ]]; then
            # If so, extract the reference column
            cut -f2 $eval > $ref
        else
            echo "File $eval not found"
        fi
    fi

    if [[ ! -f $src ]]; then
        echo "Source file $src not found, create it"
        # First check if the file exists in the data folder
        if [[ -f $eval ]]; then
            # If so, extract the source column
            cut -f1 $eval > $src
        else
            echo "File $eval not found"
        fi
    fi



    if [[ ! -f $out ]]; then
        echo "Output file $out not found, skip evaluation"
    else
        # NOTE: automatically get target language by last 2 chars of ref file
        # So assume it is called something like wiki.en-mt for example
        # Otherwise just manually specify it below
        
        # Skip whole BLEU/chrf section if last file already exists
        
        # if [[ -f "${out}.eval.chrfpp" ]]; then
        #     echo "Eval file already exists, skip BLEU and friends"
        # else
        # First put everything in 1 file
        sacrebleu $out -i $ref -m bleu ter chrf --chrf-word-order 2 > ${out}.eval.sacre
        # Add chrf++ to the previous file
        sacrebleu $out -i $ref -m chrf --chrf-word-order 2 >> ${out}.eval.sacre
        # Write only scores to individual files
        sacrebleu $out -i $ref -m bleu -b > ${out}.eval.bleu
        sacrebleu $out -i $ref -m ter -b > ${out}.eval.ter
        sacrebleu $out -i $ref -m chrf -b > ${out}.eval.chrf
        sacrebleu $out -i $ref -m chrf --chrf-word-order 2 -b > ${out}.eval.chrfpp
        # fi	

        # Calculate BLEURT (pretty slow)
        # If error: 
        # module load cuDNN
        # module load GLibmm
        # if [[ -f "${out}.eval.bleurt" ]]; then
        #     echo "Eval file already exists, skip BLEURT"
        # else
        srun python -m bleurt.score_files -candidate_file=${out} -reference_file=${ref} -bleurt_checkpoint $HOME/bleurt/BLEURT-20 -scores_file=${out}.eval.bleurt
        # fi

        # COMET (might not work so well for Maltese, as it is not in XLM-R)
        # if [[ -f "${out}.eval.comet" ]]; then
        #     echo "Eval file already exists, skip COMET"
        # else
        comet-score -s $src -t $out -r $ref > ${out}.eval.comet
        # fi

        ## BERT-score
        # First select the model based on the language
        # Highest scoring multi-lingual model (Maltese not in there)
        # if [[ $lang = "mt" ]]; then
        #     # This model is 15G, can take quite a while to download
        #     model="google/mt5-xl" 
        # else
        #     model="xlm-roberta-large" 
        # fi
        model="xlm-roberta-large" 
        # Now run the scoring
        # if [[ -f "${out}.eval.bertscore" ]]; then
        #     echo "Eval file already exists, skip bert-score"
        # else
        bert-score --lang $lang -m $model -r $ref -c $out > ${out}.eval.bertscore
        # fi
    fi
    
done
