#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=00:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rikvannoord@gmail.com
# Parse a file with a Marian trained NMT system
set -eu -o pipefail

# Read in arguments
file=$1 # File to parse
folder=$2 # Folder with the model and vocabs
out=$3 # Name we give the output file (not full folder, just the name)
config="config/parse.yml" # Config yaml file with experimental settings

# Create folder for output if needed
mkdir -p ${folder}/output

# Only do the parsing if the output file does not exist yet

if [[ -f "${folder}/output/${out}" ]]; then
	echo "Output file ${folder}/output/${out} already exists, skip parsing"
else
	# Do decoding with specified model
	# Tab separated input does not work for some reason, so always use just the input
	marian-dev/build/marian-decoder -i $file -o ${folder}/output/${out} --models ${folder}/model/model.npz -v ${folder}/vocab/vocab.spm ${folder}/vocab/vocab.spm --config $config
fi
