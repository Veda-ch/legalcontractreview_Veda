"""
Fine-tune LegalBERT on a curated subset of 20 risk-relevant
original LEDGAR clause categories.

The original LEDGAR dataset contains 100 clause categories.
This project uses 20 selected categories based on:

1. Legal and financial risk relevance
2. Importance during contract review
3. Usefulness for downstream risk analysis

The selected categories retain their original LEDGAR names.
They are not mapped into the project's previous 7 broad categories.

Run:

    python train_classifier.py

The trained model and tokenizer are saved in:

    legalbert_finetuned/
"""

from __future__ import annotations

import json
import numpy as np

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
# SELECTED ORIGINAL LEDGAR CATEGORIES
# ============================================================

SELECTED_CATEGORIES = [
    # Financial obligations and exposure
    "Payments",
    "Fees",
    "Expenses",

    # Liability and legal exposure
    "Indemnifications",
    "Indemnity",
    "Remedies",
    "Warranties",
    "Representations",

    # Contract lifecycle
    "Terminations",
    "Survival",
    "Effective Dates",

    # Information and intellectual-property risk
    "Confidentiality",
    "Disclosures",
    "Intellectual Property",

    # Regulatory and compliance risk
    "Compliance With Laws",
    "Anti-Corruption Laws",
    "Sanctions",

    # Dispute-resolution and jurisdiction risk
    "Governing Laws",
    "Arbitration",
    "Jurisdictions",
]


# ============================================================
# TRAINING SETTINGS
# ============================================================

# Use at most this many examples from each selected category.
MAX_EXAMPLES_PER_CLASS = 1000

# Maximum number of tokens from each clause.
MAX_LENGTH = 256

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

    # All original LEDGAR category names.
    original_label_names = raw.features["label"].names

    print(
        "Number of original LEDGAR categories: "
        f"{len(original_label_names)}"
    )

    print(
        "Number of selected LEDGAR categories: "
        f"{len(SELECTED_CATEGORIES)}"
    )

    # Verify that every selected category exists in LEDGAR.
    missing_categories = [
        category
        for category in SELECTED_CATEGORIES
        if category not in original_label_names
    ]

    if missing_categories:

        raise ValueError(
            "\nThe following selected categories "
            "were not found in LEDGAR:\n"
            f"{missing_categories}\n"
        )

    # Create new model output IDs from 0 to 19.
    #
    # These are not the original LEDGAR numeric IDs.
    # The original LEDGAR category names are preserved,
    # but the selected categories are remapped to a
    # compact 20-class output space.
    label2id = {
        category: index
        for index, category
        in enumerate(SELECTED_CATEGORIES)
    }

    id2label = {
        index: category
        for category, index
        in label2id.items()
    }

    texts = []
    labels = []

    # Count examples used for each selected category.
    per_class_count = {
        category: 0
        for category
        in SELECTED_CATEGORIES
    }

    print(
        "\nFiltering LEDGAR to the selected "
        "risk-relevant categories..."
    )

    for example in raw:

        text = example["text"].strip()

        if not text:
            continue

        original_label_id = example["label"]

        original_label_name = (
            original_label_names[
                original_label_id
            ]
        )

        # Ignore the other LEDGAR categories.
        if (
            original_label_name
            not in SELECTED_CATEGORIES
        ):
            continue

        # Limit the number of examples per class.
        if (
            per_class_count[
                original_label_name
            ]
            >= MAX_EXAMPLES_PER_CLASS
        ):
            continue

        texts.append(text)

        # Remap the selected category to 0–19.
        labels.append(
            label2id[
                original_label_name
            ]
        )

        per_class_count[
            original_label_name
        ] += 1

    print(
        "\nTraining examples per selected "
        "LEDGAR category:"
    )

    for category in SELECTED_CATEGORIES:

        print(
            f"{category}: "
            f"{per_class_count[category]}"
        )

    print(
        "\nTotal examples used: "
        f"{len(texts)}"
    )

    # Ensure every selected category has data.
    empty_categories = [
        category
        for category, count
        in per_class_count.items()
        if count == 0
    ]

    if empty_categories:

        raise ValueError(
            "\nNo training examples were found "
            "for:\n"
            f"{empty_categories}\n"
        )

    return (
        texts,
        labels,
        label2id,
        id2label,
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    (
        texts,
        labels,
        label2id,
        id2label,
    ) = load_ledgar()

    print(
        "\nLoading LegalBERT tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

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
            num_classes=len(
                SELECTED_CATEGORIES
            ),
            names=SELECTED_CATEGORIES,
        )
    )

    print(
        "Creating train and validation split..."
    )

    dataset = dataset.train_test_split(
        test_size=0.15,
        seed=42,
        stratify_by_column="label",
    )

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print(
        "Tokenizing the dataset..."
    )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    print(
        "Loading LegalBERT "
        "classification model..."
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(
                SELECTED_CATEGORIES
            ),
            id2label=id2label,
            label2id=label2id,
        )
    )

    def compute_metrics(
        eval_prediction
    ):

        logits, true_labels = (
            eval_prediction
        )

        predicted_labels = (
            np.argmax(
                logits,
                axis=-1,
            )
        )

        return {
            "accuracy": accuracy_score(
                true_labels,
                predicted_labels,
            ),

            "f1_macro": f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            ),
        }

    # Two epochs instead of three to reduce CPU
    # training time while still fine-tuning LegalBERT.
    training_arguments = (
        TrainingArguments(

            output_dir=(
                "./training_output"
            ),

            num_train_epochs=2,

            per_device_train_batch_size=8,

            per_device_eval_batch_size=16,

            learning_rate=2e-5,

            weight_decay=0.01,

            eval_strategy="epoch",

            save_strategy="epoch",

            load_best_model_at_end=True,

            metric_for_best_model=(
                "f1_macro"
            ),

            greater_is_better=True,

            logging_steps=50,

            save_total_limit=2,

            report_to="none",
        )
    )

    trainer = Trainer(

        model=model,

        args=training_arguments,

        train_dataset=(
            tokenized_dataset["train"]
        ),

        eval_dataset=(
            tokenized_dataset["test"]
        ),

        data_collator=(
            DataCollatorWithPadding(
                tokenizer=tokenizer
            )
        ),

        compute_metrics=(
            compute_metrics
        ),
    )

    print(
        "\nStarting LegalBERT "
        "fine-tuning..."
    )

    trainer.train()

    print(
        "\nEvaluating the "
        "fine-tuned model..."
    )

    metrics = trainer.evaluate()

    print(
        "\nFinal validation metrics:"
    )

    for (
        metric_name,
        metric_value,
    ) in metrics.items():

        print(
            f"{metric_name}: "
            f"{metric_value}"
        )

    print(
        "\nSaving fine-tuned model to "
        f"{FINETUNED_DIR}/ ..."
    )

    trainer.save_model(
        FINETUNED_DIR
    )

    tokenizer.save_pretrained(
        FINETUNED_DIR
    )

    # Save the selected category mappings.
    with open(
        (
            f"{FINETUNED_DIR}/"
            "label_mappings.json"
        ),
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "selected_categories": (
                    SELECTED_CATEGORIES
                ),

                "label2id": (
                    label2id
                ),

                "id2label": {
                    str(key): value
                    for (
                        key,
                        value,
                    )
                    in id2label.items()
                },
            },
            file,
            indent=4,
        )

    # Save final evaluation metrics.
    with open(
        (
            f"{FINETUNED_DIR}/"
            "metrics.json"
        ),
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        "\nTraining completed "
        "successfully."
    )

    print(
        "Fine-tuned model saved to: "
        f"{FINETUNED_DIR}/"
    )

    print(
        "\nThe model was trained on "
        f"{len(SELECTED_CATEGORIES)} "
        "selected original LEDGAR "
        "categories."
    )


if __name__ == "__main__":

    main()

# """Fine-tunes LegalBERT into a real classification head over our 7-category
# clause taxonomy, trained on LEDGAR (mapped down to our labels via
# label_mapping.py).

# Run once, standalone, BEFORE running the pipeline (needs internet, to pull
# LEDGAR + LegalBERT; much faster with a GPU but works on CPU too, just
# slower):

#     python inspect_ledgar_labels.py     # verify label_mapping.py first
#     python train_classifier.py

# Saves the fine-tuned model to ./legalbert_finetuned/. classifier.py checks
# for that folder automatically on the next `python pipeline.py` run and uses
# it if present — falling back to the zero-shot prototype method (embeddings.py
# + hand-written exemplars) if you haven't trained yet, so the pipeline never
# breaks either way.
# """
# from __future__ import annotations
# import numpy as np
# from datasets import load_dataset, Dataset
# from transformers import (
#     AutoTokenizer, AutoModelForSequenceClassification,
#     TrainingArguments, Trainer, DataCollatorWithPadding,
# )
# from sklearn.metrics import accuracy_score, f1_score

# from embeddings import MODEL_NAME
# from label_mapping import map_label
# from classifier import CLAUSE_TYPES, FINETUNED_DIR

# # LEDGAR has ~80k examples; cap per class so CPU training finishes in
# # reasonable time. Raise this if you have a GPU and want more data.
# MAX_EXAMPLES_PER_CLASS = 1500


# def load_and_map():
#     raw = load_dataset("lex_glue", "ledgar", split="train")
#     ledgar_names = raw.features["label"].names
#     label2id = {name: i for i, name in enumerate(CLAUSE_TYPES)}

#     texts, labels = [], []
#     per_class_count = {t: 0 for t in CLAUSE_TYPES}
#     for ex in raw:
#         fine_label = ledgar_names[ex["label"]]
#         coarse = map_label(fine_label)
#         if per_class_count[coarse] >= MAX_EXAMPLES_PER_CLASS:
#             continue
#         texts.append(ex["text"])
#         labels.append(label2id[coarse])
#         per_class_count[coarse] += 1

#     print("Training examples per class:", per_class_count)
#     if any(c < 50 for c in per_class_count.values()):
#         print("WARNING: some classes have very few examples — check "
#               "label_mapping.py against inspect_ledgar_labels.py output; "
#               "you may be missing LEDGAR label names for that category.")
#     return texts, labels, label2id


# def main():
#     texts, labels, label2id = load_and_map()
#     id2label = {i: t for t, i in label2id.items()}

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#     ds = Dataset.from_dict({"text": texts, "label": labels})
#     ds = ds.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")

#     def tokenize(batch):
#         return tokenizer(batch["text"], truncation=True, max_length=256)

#     ds = ds.map(tokenize, batched=True)

#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_NAME, num_labels=len(CLAUSE_TYPES), id2label=id2label, label2id=label2id,
#     )

#     def compute_metrics(eval_pred):
#         logits, refs = eval_pred
#         preds = np.argmax(logits, axis=-1)
#         return {
#             "accuracy": accuracy_score(refs, preds),
#             "f1_macro": f1_score(refs, preds, average="macro"),
#         }

#     args = TrainingArguments(
#         output_dir="./_train_tmp",
#         num_train_epochs=3,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=32,
#         learning_rate=2e-5,
#         weight_decay=0.01,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         load_best_model_at_end=True,
#         metric_for_best_model="f1_macro",
#         logging_steps=50,
#     )

#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=ds["train"],
#         eval_dataset=ds["test"],
#         data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
#         compute_metrics=compute_metrics,
#     )

#     trainer.train()
#     metrics = trainer.evaluate()
#     print("Final validation metrics (quote these to ma'am/the panel):", metrics)

#     trainer.save_model(FINETUNED_DIR)
#     tokenizer.save_pretrained(FINETUNED_DIR)
#     print(f"Saved fine-tuned model to {FINETUNED_DIR}/ — pipeline.py will pick it up automatically.")


# if __name__ == "__main__":
#     main()
