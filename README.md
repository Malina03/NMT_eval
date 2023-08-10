# Readme for NMT training and evaluation

Overview of training NMT systems in the **MaCoCu project**. Originally forked from [NMT_eval](https://github.com/RikVN/NMT_eval), but this project uses the PyTorch implementation of Marian models, and the Huggingface API for fine-tuning pre-trained models. For training from scratch, refer to the original repository.


## Setting up

When using the RUG Habrok cluster first load these modules:

```
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
```



Clone the repo:

```
git clone https://github.com/Malina03/NMT_eval
cd NMT_eval
```

Setup a virtual environment and use the nmt_req.txt file to install the dependencies. If you're not using Habrok, you might have to install some packages yourself.

```
python -m venv nmt
source nmt/bin/activate
pip install nmt_req.txt
```


## Data

We assume you have your data in a tab separated file with first the source and then the target in $FILE. You can use the following command to remove the duplicated lines between two or more files. This script will remove the lines in $FILE1, $FILE2, etc. that appear in the $MAIN file.   

```
python src/multiple_near_dedup.py --main_file $MAIN --other_files $FILE1 $FILE2 --dedup either 
```

There is also a slurm jobscript for this: job_scripts/near_dedup.sh. 

Keep in mind that for some (multilingual) models, you might need to add language tokens on the source side to indicate the target language variation i.e Cyrillic or Latin script. 
 
## Training

We're using the Huggingface API to fine-tune Opus-mt models. You can have a look at the possible arguments in src/utils.py. You will need to provide at least the following arguments and run the code as follows.

```
python src/train.py \
    --root_dir $root_dir \
    --train_file $train_file \
    --dev_file $dev_file \
    --exp_type fine_tune\
    --model_name $model \
    &> $log_file

```

Where the root_dir is the directory where you want your models to be saved. It ideally also contains your data and train and dev files, but you will have to provide **absolute paths** to these as well. The code will create a models directory, with sub-folders named after the exp_type and training cropus (assuming the corpus name is the first segment after splitting the train file name by ".") where the best model checkpoint is saved. The model name should correspond to the models listed on Huggingface, for instance Helsinki-NLP/opus-mt-en-sla. It's good to write the training output to a log file, to keep track of the training metrics.
```
|- root_dir
    |- data
        |- $train_file
        |- $dev_file
    |- models
        |- $exp_type
            |- corpus
                |- checkpoint

```
If you're using the already pre-processed data on Habrok, there are job scripts that can handle training. They will automatically use the correct models, training and dev files. They will also create a log file in a new logs directory, under the root_dir. You need to specify the training corpus and target language, for instance:

```
sbatch job_scripts/fine_tune.sh MaCoCuV1 bg
```
But check that the training corpus and language combination is valid, as there are no checks for that. 

There is also a separate script for the [en-hbs experiment](job_scripts/fine_tune_en-hbs.sh).


### Parsing

If you trained a model, you can use it to parse new files. You will need to provide the following args:

```
python src/train.py \
    --root_dir $root_dir \
    --train_file $test_file \
    --dev_file $test_file \
    --test_file $test_file\
    --exp_type "eval/${train_corpus}" \
    --checkpoint $checkpoint \
    --model_name $model \
    --eval \
    --predict \
    &> $log_file 
```

The Huggingface Trainer requires train and dev files, but you can pass the test file again, since it will be loaded faster than the actual training data and the model will only be tested, not trained. If you only want the evaluation metrics, then the --eval flag should be enough and the metrics will be printed in the log_file. If you also want the predictions, you need the --predict flag as well and the --exp_type will indicate where your predictions file will be saved. For instance if you're testing a model trained on MaCoCuV1, on flores_devtest, use the following:

```
python src/train.py \
    --root_dir $root_dir \
    --train_file $root_dir/flores_devtest \
    --dev_file $root_dir/flores_devtest \
    --test_file $root_dir/flores_devtest \
    --exp_type "eval/MaCoCuV1" \
    --checkpoint $root_dir/models/fine_tune/MaCoCuV1/checkpoint-* \
    --model_name $model \
    --eval \
    --predict \
    &> $log_file 
```

Then the predictions will be saved in root_dir/logs/eval/MaCoCuV1/flores_devtest_predictions.txt. 

The log_file acts as a sanity check if needed, recording the model checkpoint and testing file used, it will also show the testing mestrics (BLEU, loss, chrF,TER) as computed using the Huggingface API. You can use the ``eval.sh`` script to further test the predictions of the models.

The jobscripts predict_all.sh will test the fine-tuned models for all languages trained on a given corpus - which you can modify directly in the files. They will use the correct testing files from Habrok for all languages. The predict_baseline.sh will test the baseline models from Huggingface. The predict_hbs.sh will use the model trained on the hbs languages and test them on the test files for each language, saving the results in en-hbs/logs/eval/MaCoCuV2/${test_language}/flores_devtest_predictions.txt.

### Evaluation

The parsed files can be evaluated using ``src/eval.sh``. It will calculate BLEU, CHRF, TER, BLEURT, COMET and BERT-Score. The scores will be automatically saved in ${outfile_name}.eval.{metric}.

On Habrok, you will need the Tensorflow module instead of the PyTorch one:

```
module purge
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
```

Then you can create a new venv:

```
python -m venv eval
source eval/bin/activate
pip install req_eval.txt
```


For BLEURT, we also have to download the model (2GB):

```
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
```
I got errors before I moved the model in the bleurt folder, so you might want to do that as well after unzipping:

```
mv BLEURT-20 bleurt/BLEURT-20
```


The eval script takes three arguments as input: the translated file, the gold standard reference, and the source file (needed for COMET).

Important: the script assumes the last two letters of the gold standard are the language ISO code (e.g. the file is called dev.bg). This matters for BERT-score.

```
./src/eval.sh $out_file $gold_standard $source
```

If you're using the project set-up on Habrok, you can directly use the eval_all.sh and eval_hbs.sh files.  

The script ``src/eval_all.sh`` will automatically evaluate all predictions files for all languages trained on a particular corpus. It will find the correct prediction files, according to the training and prediction pipeline, will find the test files for each language, will make the source and reference files (if needed) and will save the output files in the same directory as the prediction files as ${prediction_file}.eval.{metric}. 

**NOTE** The script will not overwrite existing output files, so if you need to re-evaluate the models, either delete the output files manually or change the script.
