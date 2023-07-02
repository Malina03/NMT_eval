'''Fine-tune a pre-trained model from Huggingface on a new dataset.'''

from transformers import AutoTokenizer, AutoConfig, EarlyStoppingCallback, AutoModelForSeq2SeqLM, Seq2SeqTrainer, DataCollatorForSeq2Seq
from utils import get_args, get_train_args, load_data, compute_metrics
import wandb
from functools import partial

if __name__ == "__main__":
    
    args = get_args()
    if args.wandb:
        # only log the training process 
        wandb_name = f"{args.exp_type}_{args.train_file.split('/')[-1].split('.')[0]}"
        # Initialize wandb
        wandb.init(project="NMT_eval", name=wandb_name, config=args)

    
    # Load the data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, max_length=args.max_length, truncation=True)
    train_dataset = load_data(args.train_file, args, tokenizer=tokenizer)
    dev_dataset= load_data(args.dev_file, args, tokenizer=tokenizer)

    # Load the model
    if args.checkpoint is None:
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config)
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
        metrics=trainer.evaluate()
        print("\nInfo:\n", metrics, "\n")
    else:
        metrics = trainer.train()
        print("\nInfo:\n", metrics, "\n")


