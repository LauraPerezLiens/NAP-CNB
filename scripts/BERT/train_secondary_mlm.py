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

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(
    "/home/nap/lperez_nn/data/data_uniref50/secondary_structure_prediction"
)

DATA_DIR = BASE_DIR / "bert_dataset" / "secondary_mlm"

VOCAB_FILE = DATA_DIR / "vocab.txt"

OUTPUT_DIR = (
    "/home/nap/lperez_nn/model/secondary_bert_mlm"
)

# =========================================================
# TRAINING PARAMETERS
# =========================================================

WINDOW_SIZE= 25
MAX_LEN = 30
MLM_PROBABILITY = 0.15

HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
INTERMEDIATE_SIZE = 3072

BATCH_SIZE = 32
MAX_STEPS= 500_000

# =========================================================

def main():

    print("\n[INFO] Loading tokenizer...")
    print(VOCAB_FILE)

    tokenizer = BertTokenizerFast(
        vocab_file=str(VOCAB_FILE),
        do_lower_case=False,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )

    print("\n[INFO] Loading parquet datasets...")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": str(DATA_DIR / "train.parquet"),
            "validation": str(DATA_DIR / "val.parquet"),
        },
    )

    print(dataset)

    # =====================================================
    # TOKENIZATION
    # =====================================================

    def crop_secondary_structure(ss_spaced):
            ss = ss_spaced.replace(" ", "").strip()

            if len(ss) > WINDOW_SIZE:
                ss = ss[:WINDOW_SIZE]

            return " ".join(ss)

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

    print("\n[INFO] Tokenizing dataset...")

    tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=8,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing secondary structure sequences",
        )


    # =====================================================
    # MODEL CONFIG
    # =====================================================

    print("\n[INFO] Building BERT config...")

 
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

    print(f"\n[INFO] Model parameters: {total_params:,}")

    # =====================================================
    # MLM COLLATOR
    # =====================================================

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    # =====================================================
    # TRAINING ARGS
    # =====================================================

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        max_steps=MAX_STEPS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,

        logging_steps=500,

        eval_strategy="steps",
        eval_steps=5000,

        save_strategy="steps",
        save_steps=5000,

        save_total_limit=3,

        fp16=True,

        dataloader_num_workers=8,

        report_to="none",

        remove_unused_columns=False,
    )


    # =====================================================
    # TRAINER
    # =====================================================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    # =====================================================
    # TRAIN
    # =====================================================

    print("\n[INFO] Starting MLM training...\n")

    trainer.train()

    # =====================================================
    # SAVE
    # =====================================================

    print("\n[INFO] Saving model...")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n[OK] Model saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()