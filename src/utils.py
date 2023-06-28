import argparse
from transformers import Seq2SeqTrainingArguments, AutoTokenizer
import evaluate
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-root_dir", "--root_dir", required=True, type=str, help="Root directory.")
    # parser.add_argument("-logging_dir", "--logging_dir", required=False, type=str, default="...", help="Logging directory.")
    # parser.add_argument("-model_save_dir", "--model_save_dir", required=True, type=str, help="Path to the output directory where the model will be saved.")
    parser.add_argument("checkpoint", "--checkpoint", required=False, type=str, help="Path to the checkpoint to fine-tune. If not provided, the model will be initialized from scratch.")
    parser.add_argument("-eval", "--eval", required=False, action="store_true", help="Whether to only evaluate the model.")
    parser.add_argument("-exp_type", "--exp_type", required=False, type=str, default="fine_tuning", help="Type of experiment. Can be 'fine_tuning' or 'from_scratch'.")
    parser.add_argument("-wandb", "--wandb", required=False, action="store_true", help="Whether to log the training process on wandb.")

    parser.add_argument("-train_file", "--train_file", required=True, type=str, help="Path to the training file.")
    parser.add_argument("-dev_file", "--dev_file", required=True, type=str, help="Path to the development data  file.")
    parser.add_argument("-test_file", "--test_file", required=False, type=str, help="Path to the test data file.")

    parser.add_argument("-model_name", "--model_name", required=True, type=str, help="Name of the model to fine-tune. Must be a model from Huggingface.")
    parser.add_argument("-max_length", "--max_length", required=False, type=int, default=512, help="Maximum length of the input sequence.")
   
    parser.add_argument("-seed", "--seed", required=False, type=int, default=1, help="Random seed.")
    parser.add_argument("-num_train_epochs", "--num_train_epochs", required=False, type=int, default=10, help="Number of training epochs.")
    parser.add_argument("-batch_size", "--batch_size", required=False, type=int, default=32, help="Batch size.")
    parser.add_argument("-metric_for_best_model", "--metric_for_best_model", required=False, type=str, default="bleu", help="Metric to use to select the best model.")
    parser.add_argument("-evaluation_strategy", "--evaluation_strategy", required=False, type=str, default="epoch", help="Strategy to adopt for evaluation during training.")
    parser.add_argument("-save_strategy", "--save_strategy", required=False, type=str, default="epoch", help="Strategy to adopt for saving checkpoints during training.")
    parser.add_argument("-learning_rate", "--learning_rate", required=False, type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("-max_grad_norm", "--max_grad_norm", required=False, type=float, default=1, help="Maximum gradient norm.")
    parser.add_argument("-warmup_steps", "--warmup_steps", required=False, type=int, default=200, help="Number of warmup steps.")
    parser.add_argument("-weight_decay", "--weight_decay", required=False, type=float, default=0, help="Weight decay.")
    parser.add_argument("-logging_steps", "--logging_steps", required=False, type=int, default=1000, help="Logging steps.")
    parser.add_argument("-evaluation_steps", "--evaluation_steps", required=False, type=int, default=1000, help="Evaluation steps.")
    parser.add_argument("-save_total_limit", "--save_total_limit", required=False, type=int, default=1, help="Maximum number of checkpoints to save.")
    parser.add_argument("-save_steps", "--save_steps", required=False, type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument("-early_stopping_patience", "--early_stopping", required=False, type=int, default=3, help="Early stopping patience.")
    parser.add_argument("label_smoothing", "--label_smoothing", required=False, type=float, default=0.1, help="Label smoothing.")
    parser.add_argument("-dropout", "--dropout", required=False, type=float, default=0.1, help="Dropout.")
    args = parser.parse_args()
    return args

def get_train_args(args):
    model_save_dir = os.path.join(args.root_dir, "models", args.model_name, args.exp_type)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    logging_dir = os.path.join(args.root_dir, "logs", args.model_name, args.exp_type)
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    train_args = Seq2SeqTrainingArguments(
        output_dir=model_save_dir,
        logging_dir=logging_dir,
        logging_steps=args.logging_steps,
        evaluation_steps=args.evaluation_steps,
        save_steps=args.save_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size *2,
        metric_for_best_model=args.metric_for_best_model,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        seed=args.seed,
        load_best_model_at_end=True,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        label_smoothing_factor=args.label_smoothing,
        early_stopping_patience=args.early_stopping,
        predict_with_generate=True,
        report_to="wandb" if args.wandb else "none",
    )
    return train_args

def load_data(file, args):
    # Load the data
    corpus_src = []
    corpus_tgt = []
    source_lang = args.file.split('.')[1].split('-')[0]
    target_lang = args.file.split('.')[1].split('-')[1]
    prefix = ""
    with open(file, 'r', encoding="utf-8") as f:
        for line in f:
            src, tgt = line.strip().split('\t')
            corpus_src.append(src)
            corpus_tgt.append(tgt)
    inputs = [prefix + src for src in corpus_src]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, max_length=args.max_length, truncation=True, padding=True)
    model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True, padding=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(corpus_tgt, max_length=args.max_length, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
            
def compute_metrics(preds):
    labels_ids = preds.label_ids
    if isinstance(preds.predictions, tuple):
        preds_ids = preds.predictions[0]
    else:
        preds_ids = preds.predictions
    preds_ids = preds.predictions.argmax(-1)
    decode_preds = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    decode_labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    decode_preds = [pred.strip() for pred in decode_preds]
    decode_labels = [label.strip() for label in decode_labels]
    results = {}
    chrf = evaluate.load("chrf")
    bleu = evaluate.load("bleu")
    results["bleu"] = bleu.compute(predictions=decode_preds, references=decode_labels)
    results["chrf"] = chrf.compute(predictions=decode_preds, references=decode_labels)
    return results

