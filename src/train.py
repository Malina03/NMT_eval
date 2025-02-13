'''Fine-tune a pre-trained model from Huggingface on a new dataset.'''

from transformers import AutoTokenizer, EarlyStoppingCallback, AutoModelForSeq2SeqLM, Seq2SeqTrainer, DataCollatorForSeq2Seq
from utils import get_args, get_train_args, load_data, compute_metrics
import wandb
from functools import partial
import os

if __name__ == "__main__":
    
    args = get_args()
    if args.wandb:
        # only log the training process 
        wandb_name = f"{args.train_file.split('/')[-1].split('.')[1]}_{args.train_file.split('/')[-1].split('.')[0]}"
        # Initialize wandb
        wandb.init(project="NMT_eval", name=wandb_name, config=args)

    
    # Load the data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, max_length=args.max_length, truncation=True)
    train_dataset = load_data(args.train_file, args, tokenizer=tokenizer)
    dev_dataset= load_data(args.dev_file, args, tokenizer=tokenizer)

    if args.eval or args.predict:
        test_dataset = load_data(args.test_file, args, tokenizer=tokenizer)

 
    # Load the model
    if args.checkpoint is None:
        # config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, local_files_only=True)

    # Set the training arguments
    training_args = get_train_args(args)

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping)
    ]
    
    # Instantiate the trainer
    trainer =  Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        callbacks=callbacks
    )

    if args.eval:
        if args.predict:
            output = trainer.predict(test_dataset=test_dataset)
            preds = output.predictions
            if isinstance(preds, tuple):
                preds = preds[0]
            decode_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            predictions = [pred.strip() for pred in decode_preds]
            logging_dir = os.path.join(args.root_dir, "logs", args.exp_type)
            if not os.path.exists(logging_dir):
                os.makedirs(logging_dir)
            eval_corpus = args.test_file.split("/")[-1].split(".")[0]
            with open(os.path.join(logging_dir, f'{eval_corpus}_predictions.txt'), "w") as f:
                for pred in predictions:
                    f.write(pred + "\n")
            print("\nInfo:\n", output.metrics, "\n")
            print("Tested on:", args.test_file)
            print('Predictions saved to:', os.path.join(logging_dir, f'{eval_corpus}_predictions.txt'))
            if args.checkpoint is not None:
                print("Model from:", args.checkpoint)
            else:
                print("Baseline model:", args.model_name)
     
        else:
            metrics = trainer.evaluate()
            print("\nInfo:\n", metrics, "\n")

    else:
        ## evaluate the baseline model before training
        if args.eval_baseline:
            metrics = trainer.evaluate()
            print("\nBaseline metrics:\n", metrics, "\n")
        metrics = trainer.train()
        print("\nInfo:\n", metrics, "\n")


