"""
Fine-tune LegalBERT on all 100 original LEDGAR clause categories.

The classifier labels are the ORIGINAL LEDGAR category names, unmapped and
uncollapsed. CUAD and MAUD are NOT used to reduce or relabel these
categories — they remain part of the separate legal knowledge base used
during retrieval (see build_knowledge_base.py), not the classifier.

Note: fine-tuning all 100 categories for 3 epochs is impractical on a
CPU-only machine (30+ hours estimated). This has been run and verified on a
free Google Colab T4 GPU (~30 minutes) instead; run it there if you don't
have local GPU access — Runtime -> Change runtime type -> T4 GPU.

Run:

    python train_classifier.py

The trained model and tokenizer are saved in:

    legalbert_finetuned/
"""

from __future__ import annotations

import json
import numpy as np
import torch

from datasets import ClassLabel, Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from embeddings import MODEL_NAME


# ============================================================
# TRAINING SETTINGS
# ============================================================

# Use at most this many examples from each of LEDGAR's 100 categories.
# 1000 x 100 = up to 100,000 examples (fewer in practice — not every
# category has 1000 examples available).
MAX_EXAMPLES_PER_CLASS = 1000

# Maximum number of tokens from each clause.
MAX_LENGTH = 256

# Training epochs.
NUM_EPOCHS = 3

# Validation split proportion.
VALIDATION_SIZE = 0.15

# Random seed for reproducibility.
RANDOM_SEED = 42

# Folder where the fine-tuned model will be saved.
FINETUNED_DIR = "legalbert_finetuned"


# ============================================================
# LOAD AND FILTER LEDGAR
# ============================================================

def load_ledgar():

    print("Loading LEDGAR dataset...")

    raw = load_dataset(
        "lex_glue",
        "ledgar",
        split="train"
    )

    # All 100 original LEDGAR category names, used as-is.
    original_label_names = raw.features["label"].names

    print(f"Number of LEDGAR categories: {len(original_label_names)}")

    label2id = {
        category: index
        for index, category in enumerate(original_label_names)
    }

    id2label = {
        index: category
        for category, index in label2id.items()
    }

    texts = []
    labels = []

    # Count examples used for each category.
    per_class_count = {category: 0 for category in original_label_names}

    print("\nFiltering LEDGAR (capping at "
          f"{MAX_EXAMPLES_PER_CLASS} examples per category)...")

    for example in raw:

        text = example["text"].strip()

        if not text:
            continue

        label_name = original_label_names[example["label"]]

        if per_class_count[label_name] >= MAX_EXAMPLES_PER_CLASS:
            continue

        texts.append(text)
        labels.append(label2id[label_name])
        per_class_count[label_name] += 1

    print("\nExamples per category:")
    for category in original_label_names:
        print(f"{category}: {per_class_count[category]}")

    print(f"\nTotal examples used: {len(texts)}")

    empty_categories = [
        category for category, count in per_class_count.items() if count == 0
    ]

    if empty_categories:
        raise ValueError(
            f"\nNo training examples were found for:\n{empty_categories}\n"
        )

    print("\nAll 100 LEDGAR categories contain at least one training example.")

    return texts, labels, label2id, id2label


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)

    if torch.cuda.is_available():
        print("GPU detected:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected — this will be slow (30+ hours for all 100 "
              "categories / 3 epochs). Consider running on a GPU runtime "
              "(e.g. Google Colab) instead.")

    (
        texts,
        labels,
        label2id,
        id2label,
    ) = load_ledgar()

    print("\nLoading LegalBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = Dataset.from_dict(
        {
            "text": texts,
            "label": labels,
        }
    )

    # Convert the integer label column into ClassLabel.
    # This is required because stratify_by_column only
    # supports ClassLabel columns.
    dataset = dataset.cast_column(
        "label",
        ClassLabel(
            num_classes=len(label2id),
            names=list(label2id.keys()),
        )
    )

    print("Creating train and validation split...")

    dataset = dataset.train_test_split(
        test_size=VALIDATION_SIZE,
        seed=RANDOM_SEED,
        stratify_by_column="label",
    )

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("Tokenizing the dataset...")

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    print("Loading LegalBERT classification model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_prediction):

        logits, true_labels = eval_prediction

        predicted_labels = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "f1_macro": f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            ),
        }

    training_arguments = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"\nStarting LegalBERT fine-tuning — all {len(label2id)} LEDGAR categories...")

    trainer.train()

    print("\nEvaluating the fine-tuned model...")

    metrics = trainer.evaluate()

    print("\nFinal validation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    print(f"\nSaving fine-tuned model to {FINETUNED_DIR}/ ...")

    trainer.save_model(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)

    with open(f"{FINETUNED_DIR}/label_mappings.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "categories": list(label2id.keys()),
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            file,
            indent=4,
        )

    with open(f"{FINETUNED_DIR}/metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nTraining completed successfully.")
    print(f"Fine-tuned model saved to: {FINETUNED_DIR}/")
    print(f"\nThe model was trained on all {len(label2id)} original LEDGAR categories.")


if __name__ == "__main__":
    main()
