import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =========================
# CONFIG
# =========================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = "data/processed/instructions.jsonl"
OUTPUT_DIR = "outputs/mistral_lora"

MAX_LENGTH = 512
BATCH_SIZE = 2
LR = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.1
SAVE_EPOCH_FRACTION = 0.25

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05


# =========================
# DATASET
# =========================
def load_instruction_dataset(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompt = (
                "### Instruction:\n"
                f"{obj['instruction']}\n\n"
                "### Response:\n"
                f"{json.dumps(obj['response'])}"
            )
            records.append({"text": prompt})

    return Dataset.from_list(records)


# =========================
# TOKENIZATION
# =========================
def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    # =========================
    # LORA CONFIG (ALL LINEAR)
    # =========================
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    dataset = load_instruction_dataset(DATA_PATH)
    dataset = dataset.map(tokenize_fn, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    steps_per_epoch = len(dataset) // BATCH_SIZE
    save_steps = int(steps_per_epoch * SAVE_EPOCH_FRACTION)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=50,
        save_steps=save_steps,
        save_total_limit=10,
        evaluation_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    print("Saving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    print("Training complete.")
