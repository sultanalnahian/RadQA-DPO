import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import DPOTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import pandas as pd
import numpy as np



# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.5, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="models/sft/",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=8, metadata={"help": "eval batch size per device"})
    train_epoch: Optional[int] = field(default=8, metadata={"help": "number of training epoch"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    max_prompt_length: Optional[int] = field(default=768, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_target_length: Optional[int] = field(default=128, metadata={"help": "the maximum length of target sequence."})
    max_steps: Optional[int] = field(default=10000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "the evaluation frequency"})

    checkpoint_dir: Optional[str] = field(default="checkpoints/dpo", metadata={"help": "directory to save checkpoints"})
    output_dir: Optional[str] = field(default="models/dpo", metadata={"help": "the output directory"})
    train_file: Optional[str] = field(default="dataset/preference_dataset/t5-3b/train_preference_90.tsv", metadata={"help": "the training file"})
    validation_file: Optional[str] = field(default="dataset/preference_dataset/t5-3b/dev_preference.tsv", metadata={"help": "the validation file"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    

def load_data(data_files):
    data = pd.read_csv(open(data_files,'r'), delimiter="\t")
    data = data.sample(frac=1)
    return data

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name_or_path, device_map="auto")
    model_ref = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name_or_path, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    
    train_data = load_data(script_args.train_file)
    eval_data = load_data(script_args.validation_file)
    
    train_dataset = Dataset.from_pandas(train_data)
    eval_dataset = Dataset.from_pandas(eval_data)

    train_dataset = train_dataset.filter(
        lambda x: len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) <= script_args.max_length
        and len(tokenizer(x["prompt"] + x["rejected"])["input_ids"]) <= script_args.max_length
    )

    # 3. Load evaluation dataset
    eval_dataset = eval_dataset.filter(
        lambda x: len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) <= script_args.max_length
        and len(tokenizer(x["prompt"] + x["rejected"])["input_ids"]) <= script_args.max_length
    )

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="epoch",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.checkpoint_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        save_strategy = "epoch",
        num_train_epochs = script_args.train_epoch,
        # bf16=True,
        remove_unused_columns=False,
        run_name="dpo_radqa",
        seed = 42,

    )

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        # loss_type = "kto_pair"
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.checkpoint_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)