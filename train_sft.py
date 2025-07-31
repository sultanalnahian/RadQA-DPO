import torch
import argparse
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, EarlyStoppingCallback
from radqa import RadQA
import numpy as np
from datasets import Dataset
import datasets
from sklearn.metrics import accuracy_score
from util import compute_f1_score

torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_metrics(eval_preds,_tokenizer):
    logits, labels = eval_preds
    decoded_preds = _tokenizer.batch_decode(logits, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, _tokenizer.pad_token_id)
    decoded_labels = _tokenizer.batch_decode(labels, skip_special_tokens= True)
    accuracy = accuracy_score(decoded_labels, decoded_preds)
    f1_score = compute_f1_score(decoded_preds, decoded_labels)
    return {"f1_score": f1_score}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--base_model", type=str, default="t5-3b")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft")
    parser.add_argument("--output_model", type=str, default="models/sft")
    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--valid_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--train_data_file", type=str, default="dataset/train.json")
    parser.add_argument("--validation_data_file", type=str, default="dataset/dev.json")
    parser.add_argument("--early_stopping", type=int, default=3)
    return parser.parse_args()

max_input_length = 1024
max_target_length = 128

def preprocess_function(examples, _tokenizer):
    context = examples['context']
    labels = examples['answer']
    model_inputs = _tokenizer(context, max_length=max_input_length, truncation=True)
    labels = _tokenizer(text_target=labels, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<context>', '<question>']}
    )
    tokenizer.add_tokens(["no_answer"])
    config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
    model = T5ForConditionalGeneration.from_pretrained(args.base_model)
    model.resize_token_embeddings(len(tokenizer))
    # model= nn.DataParallel(model)
    model = model.to(device)

    train_data = RadQA(args.train_data_file)
    val_data = RadQA(args.validation_data_file)
    
    train_raw_datasets = Dataset.from_list(train_data.data[:50])
    valid_raw_datasets = Dataset.from_list(val_data.data[:20])
    raw_datasets = datasets.DatasetDict({"train":train_raw_datasets, "validation":valid_raw_datasets})
    preprocess_function(raw_datasets['train'][:2], tokenizer)
    tokenized_datasets = raw_datasets.map(lambda x: preprocess_function(x, tokenizer), batched = True)
    train_tokenized_datasets = tokenized_datasets['train'].remove_columns(['context','answer'])
    valid_tokenized_datasets = tokenized_datasets['validation'].remove_columns(['context','answer'])
    
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.checkpoint,          # output directory
        num_train_epochs= args.epochs,              # total number of training epochs
        evaluation_strategy = "epoch",
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.valid_batch_size,   # batch size for evaluation
        warmup_steps=100,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        learning_rate=args.learning_rate,
        predict_with_generate=True,
        remove_unused_columns=False,
        save_strategy = "epoch",
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        seed = 42,
        load_best_model_at_end = True,
        # generation_max_length = 64,
        )   
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_tokenized_datasets,         # training dataset
        eval_dataset=valid_tokenized_datasets,             # evaluation dataset
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping)],
        )
    
    trainer.train()
    trainer.save_model(args.output_model)
    metrics = trainer.evaluate()