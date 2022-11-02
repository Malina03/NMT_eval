# Readme for NMT training and evaluation

Overview of training NMT systems in the MaCoCu project.

## Setting up

Clone the repo:

```
git clone https://github.com/RikVN/NMT_eval
cd NMT_eval
```

Setup a Conda environment:

```
conda create -n nmt python=3.8
conda activate nmt
```

If you use the RUG Peregrine cluster, always load these modules:

```
module load CUDA
module load GCC/10.2.0
module load GCCcore/10.2.0
module load CMake/3.18.4-GCCcore-10.2.0
```

For the NMT experiments, we use Marian:

```
git clone https://github.com/marian-nmt/marian-dev
mkdir marian-dev/build
cd marian-dev/build
cmake ..
make -j4
```

For cleaning the data, we use bifixer and bicleaner-ai. If you are not interested in cleaning your data this way, you can skip this step. Otherwise, install like this:

```
pip install bicleaner-ai
pip install https://github.com/kpu/kenlm/archive/master.zip --install-option="--max_order 7"
sudo apt install python-dev libhunspell-dev
pip install bifixer
```

Also check if there's an extra package available for the language you are interested in, e.g. for Bulgarian:

```
sudo apt-get install hunspell-bg
```

Yes, these need to be downloaded separately per language code.

## Data

We assume you have your data in a tab separated file with first the source and then the target in $FILE.

First we will near-deduplicate the data. If either the source or target is a (near)-duplicate, we filter it:

```
python src/neardup.py -i $FILE -d either > ${FILE}.dedup
```

Then, we will fix and clean the data using bifixer and bicleaner-ai. For this to work, you have to download the model for your language (say Bulgarian) like this:

```
mkdir -p bicleaner_models
./src/bicleaner-ai-download.sh en is lite bicleaner_models
```

Then, run the cleaning like this:

```
./src/clean.sh ${FILE}.dedup en bg
```

This applies bifixer and then runs bicleaner-ai-classify (on GPU if possible), saving it in ${FILE}.dedup.clean. We then filter all sentence pairs with a bicleaner score lower than 0.5 and save it to {FILE}.dedup.clean.filter.

If you get odd Tensorflow/Keras errors, it might help to convert back to version 2.6.5:

```
pip uninstall tensorflow
pip uninstall keras
pip install tensorflow==2.6.5
```

If you get encoding errors, maybe you have to specify that we work with text in UTF-8:

```
export LANG=en_US.UTF-8
```

## Training

Training is performed by Marian and uses configuration yaml files. In config/config.yml is an example.

**You should specify your own train/dev sets in this config file.** Perhaps you also want to change the validation and saving frequency. Or the amount of memory you want to take (differs per GPU, now set to 24GB).

The creation of the vocabulary is done automatically by applying SentencePiece to the training set. There is a shell script to make training a bit easier. You have to specify the folder to store the models/output (will be created) and the config file as arguments:

```
./src/train.sh exp/ config/config.yml
```

This will apply the final model on the dev set and save it in exp/output/dev_epoch_{E}.out. Of course training can take quite some time.

### Parsing

If you trained a model, you can use it to parse new files. If we have a file with English sentences in $EN, we can parse it like this with our model:

```
./src/parse.sh $EN exp/ output_file
```

The script takes three arguments, the name of the input file, the general model folder (so not the actual model!) and the name of the output file (in this case just "output_file"). It will automatically locate the model and vocab and use the configuration in config/parse.yml.

Since you only have to specify the output name, the file can be found in ``exp/output/output_file``.

### Evaluation

The parsed files can be evaluated using ``src/eval.sh``. It will calculate BLEU, CHRF, TER, BLEURT, COMET and BERT-Score. The scores will be automatically saved in ${outfile_name}.eval.{metric}.

There are version conflicts with bicleaner-ai for certain packages, so I suggest having a fresh environment when installing the packages:

```
conda create -n eval python=3.8
conda activate eval
pip install -r req_eval.txt
```

If you get errors, it could help to revert Tensorflow back to 2.7.0:

```
pip install tensorflow==2.7.0
```

On Peregrine (RUG), also load this:

```
module load GLibmm
```

For BLEURT, we also have to download the model (2GB):

```
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
```

The eval script takes three arguments as input: the translated file, the gold standard reference, and the source file (needed for COMET).

**Important**: the script assumes the last two letters of the gold standard are the language ISO code (e.g. the file is called dev.bg). This matters for BERT-score.

```
./src/eval.sh $out_file $gold_standard $source
```

There is also a helpful Python file for summarizing all metrics across multiple experiments. If you give it a folder called "exp/", it will look in all subfolders for output files, e.g. in ``exp/exp1/output/``, ``exp/exp2/output/``, etc. You can specify the metrics you are interested in (default the ones from eval.sh) and the names of the files that are present in the ``output`` folder, e.g. flores_dev, wmt16, TED, Wiki, etc. An example call would be this:

```
python src/summarize_eval.py -i exp/ -e flores_dev flores_devtest
```

Which should print a nice table, if you have actual results.
