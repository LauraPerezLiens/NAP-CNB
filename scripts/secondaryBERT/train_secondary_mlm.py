#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a BERT masked-language model on secondary-structure sequences.

This script trains a BERT model using C/H/E secondary-structure strings
generated from ProteinUnet predictions. The input dataset must already be
built as parquet files by build_secondary_mlm_dataset.py.

Input:
    bert_mlm_dataset/train.parquet
    bert_mlm_dataset/val.parquet
    bert_mlm_dataset/vocab.txt

Output:
    /home/nap/lperez_nn/model/secondary_bert_mlm
"""

import argparse
import logging
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


WINDOW_SIZE = 25
MAX_LEN = 30
MLM_PROBABILITY = 0.15

HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
INTERMEDIATE_SIZE = 3072

BATCH_SIZE = 32
NUM_EPOCHS = 1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05

NUM_PROC = 2
DATALOADER_NUM_WORKERS = 2


def setup_logging() -> None:
    """Configure logging format and verbosity level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train secondary-structure BERT MLM model."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/nap/lperez_nn/data/data_uniref50"),
        help="Base UniRef data directory.",
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset directory. Default: <data-dir>/bert_mlm_dataset",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/nap/lperez_nn/model/secondary_bert_mlm"),
        help="Output directory for the trained BERT MLM model.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Per-device training batch size.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate.",
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=WARMUP_RATIO,
        help="Warmup ratio used to compute warmup steps.",
    )

    parser.add_argument(
        "--mlm-probability",
        type=float,
        default=MLM_PROBABILITY,
        help="Masking probability for MLM training.",
    )

    return parser.parse_args()


def validate_inputs(dataset_dir: Path) -> None:
    """Validate required input files."""
    required_files = [
        dataset_dir / "train.parquet",
        dataset_dir / "val.parquet",
        dataset_dir / "vocab.txt",
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(f"Required input file not found: {file_path}")


def crop_secondary_structure(ss_spaced: str) -> str:
    """
    Crop a spaced secondary-structure sequence to WINDOW_SIZE residues.

    The input is expected to be a space-separated C/H/E string.
    """
    ss = ss_spaced.replace(" ", "").strip()

    if len(ss) > WINDOW_SIZE:
        ss = ss[:WINDOW_SIZE]

    return " ".join(ss)


def main() -> None:
    """Train secondary-structure BERT MLM model."""
    setup_logging()
    args = parse_args()

    dataset_dir = args.dataset_dir or args.data_dir / "bert_mlm_dataset"
    vocab_file = dataset_dir / "vocab.txt"

    validate_inputs(dataset_dir)

    logging.info("Starting secondary-structure BERT MLM training")
    logging.info("Dataset directory: %s", dataset_dir)
    logging.info("Vocabulary file: %s", vocab_file)
    logging.info("Output directory: %s", args.output_dir)

    logging.info("Loading tokenizer")
    tokenizer = BertTokenizerFast(
        vocab_file=str(vocab_file),
        do_lower_case=False,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )

    logging.info("Loading parquet datasets")
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": str(dataset_dir / "train.parquet"),
            "validation": str(dataset_dir / "val.parquet"),
        },
    )

    required_column = "secondary_structure_spaced"

    for split_name in ["train", "validation"]:
        if required_column not in dataset[split_name].column_names:
            raise ValueError(
                f"Required column '{required_column}' not found in {split_name} dataset"
            )

    logging.info("Dataset loaded: %s", dataset)
    logging.info(
        "Proteins without valid ProteinUnet predictions are already excluded "
        "from the parquet datasets."
    )

    def tokenize_function(examples):
        cropped_sequences = [
            crop_secondary_structure(ss)
            for ss in examples["secondary_structure_spaced"]
        ]

        return tokenizer(
            cropped_sequences,
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
            return_special_tokens_mask=True,
        )

    logging.info("Tokenizing dataset")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing secondary-structure sequences",
    )

    num_train_samples = len(tokenized_dataset["train"])

    num_gpus = torch.cuda.device_count()
    effective_num_devices = max(num_gpus, 1)

    effective_batch_size = args.batch_size * effective_num_devices
    steps_per_epoch = math.ceil(num_train_samples / effective_batch_size)
    total_training_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    logging.info("Training size check")
    logging.info("Train samples: %d", num_train_samples)
    logging.info("GPUs detected: %d", num_gpus)
    logging.info("Per-device batch size: %d", args.batch_size)
    logging.info("Effective batch size: %d", effective_batch_size)
    logging.info("Steps per epoch: %d", steps_per_epoch)
    logging.info("Number of epochs: %d", args.num_epochs)
    logging.info("Total expected steps: %d", total_training_steps)
    logging.info("Warmup steps: %d", warmup_steps)

    logging.info("Building BERT config")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act="gelu",
        max_position_embeddings=MAX_LEN,
        type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = BertForMaskedLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info("Model parameters: %d", total_params)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        logging_steps=500,
        eval_strategy="no",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    logging.info("Starting MLM training")
    trainer.train()

    logging.info("Saving model")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    logging.info("Model saved to: %s", args.output_dir)
    logging.info("Secondary-structure BERT MLM training completed successfully")


if __name__ == "__main__":
    main()