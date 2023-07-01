import argparse
from transformers import Seq2SeqTrainingArguments, AutoTokenizer
import evaluate
import os
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-root_dir", "--root_dir", required=True, type=str, help="Root directory.")
    # parser.add_argument("-logging_dir", "--logging_dir", required=False, type=str, default="...", help="Logging directory.")
    # parser.add_argument("-model_save_dir", "--model_save_dir", required=True, type=str, help="Path to the output directory where the model will be saved.")
    parser.add_argument("-checkpoint", "--checkpoint", required=False, type=str, help="Path to the checkpoint to fine-tune. If not provided, the model will be initialized from scratch.")
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
    parser.add_argument("-batch_size", "--batch_size", required=False, type=int, default=16, help="Batch size.")
    parser.add_argument("-gradient_accumulation_steps", "--gradient_accumulation_steps", required=False, type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("-adam_epsilon", "--adam_epsilon", required=False, type=float, default=1e-9, help="Epsilon for Adam optimizer.")
    parser.add_argument("-adam_beta1", "--adam_beta1", required=False, type=float, default=0.9, help="Beta1 for Adam optimizer.")
    parser.add_argument("-adam_beta2", "--adam_beta2", required=False, type=float, default=0.98, help="Beta2 for Adam optimizer.")
    parser.add_argument("-metric_for_best_model", "--metric_for_best_model", required=False, type=str, default="val_loss", help="Metric to use to select the best model.")
    parser.add_argument("-evaluation_strategy", "--evaluation_strategy", required=False, type=str, default="epoch", help="Strategy to adopt for evaluation during training.")
    parser.add_argument("-save_strategy", "--save_strategy", required=False, type=str, default="epoch", help="Strategy to adopt for saving checkpoints during training.")
    parser.add_argument("-learning_rate", "--learning_rate", required=False, type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("-max_grad_norm", "--max_grad_norm", required=False, type=float, default=1, help="Maximum gradient norm.")
    parser.add_argument("-warmup_steps", "--warmup_steps", required=False, type=int, default=200, help="Number of warmup steps.")
    parser.add_argument("-weight_decay", "--weight_decay", required=False, type=float, default=0, help="Weight decay.")
    parser.add_argument("-logging_steps", "--logging_steps", required=False, type=int, default=10000, help="Logging steps.")
    parser.add_argument("-evaluation_steps", "--evaluation_steps", required=False, type=int, default=10000, help="Evaluation steps.")
    parser.add_argument("-save_total_limit", "--save_total_limit", required=False, type=int, default=1, help="Maximum number of checkpoints to save.")
    parser.add_argument("-save_steps", "--save_steps", required=False, type=int, default=10000, help="Save checkpoint every X updates steps.")
    parser.add_argument("-early_stopping", "--early_stopping", required=False, type=int, default=3, help="Early stopping patience.")
    parser.add_argument("-label_smoothing", "--label_smoothing", required=False, type=float, default=0.1, help="Label smoothing.")
    args = parser.parse_args()
    return args


class HFDataset(torch.utils.data.Dataset):
    """Dataset for using HuggingFace Transformers."""

    def __init__(self, encodings, decoder_input_ids):
        self.encodings = encodings
        self.decoder_input_ids = decoder_input_ids

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.decoder_input_ids[idx])
        return item

    def __len__(self):
        return len(self.decoder_input_ids)


def get_train_args(args):
    model_save_dir = os.path.join(args.root_dir, "models", args.model_name, args.exp_type, args.train_file.split("/")[-1].split(".")[0])
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    logging_dir = os.path.join(args.root_dir, "logs", args.model_name, args.exp_type, args.train_file.split("/")[-1].split(".")[0])
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    train_args = Seq2SeqTrainingArguments(
        output_dir=model_save_dir,
        logging_dir=logging_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size *2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        metric_for_best_model=args.metric_for_best_model,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        seed=args.seed,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        load_best_model_at_end=True,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        report_to="wandb" if args.wandb else "none",
    )
    return train_args

def load_data(filename, args, tokenizer):
    # Load the data
    corpus_src = []
    corpus_tgt = []
    error_count = 0
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                src, tgt = line.strip().split('\t')
                corpus_src.append(src)
                corpus_tgt.append(tgt)
            except:
                error_count += 1
                continue
    print("Error count: ", error_count)
    model_inputs = tokenizer(corpus_src, max_length=args.max_length, truncation=True)
    encoded_tgt = tokenizer(text_target=corpus_tgt, max_length=args.max_length, truncation=True)
    return HFDataset(model_inputs, encoded_tgt["input_ids"])
            

def compute_metrics(preds):
    labels_ids = preds.label_ids
    if isinstance(preds.predictions, tuple):
        preds_ids = preds.predictions[0]
    else:
        preds_ids = preds.predictions
    preds_ids = preds.predictions.argmax(-1)
    decode_preds = tokenizer.batch_decode(preds_ids, use_source_tokenizer=True, skip_special_tokens=True)
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

