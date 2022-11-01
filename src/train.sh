#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=11:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rikvannoord@gmail.com

# Train an NMT model using Marian
set -eu -o pipefail

# Read in arguments
fol=$1 # Experiment folder where we save everything
config=$2 # Config yaml file with experimental settings

# Set up folders
mkdir -p $fol
mkdir -p ${fol}/vocab ${fol}/model ${fol}/bkp ${fol}/output

# Train model - specifying the vocab like this automatically builds it
marian-dev/build/marian -m ${fol}/model/model.npz --tsv --vocabs ${fol}/vocab/vocab.spm ${fol}/vocab/vocab.spm --config $config --valid-translation-output ${fol}/output/dev_epoch_{E}.out

# If you want to fine-tune, you should add these arguments to the training call:
#  --valid-reset-stalled --no-restore-corpus --ignore-model-config
