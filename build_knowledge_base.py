"""
Build a unified legal knowledge base from LEDGAR, CUAD, and MAUD.

Output:
    data/knowledge_base/legal_records.jsonl

Each output record contains:
    source
    source_id
    title
    category
    text
    metadata

Dataset roles
-------------
LEDGAR:
    Contract clause classification examples.

CUAD:
    Contract paragraphs linked to specific clause categories.
    The COMPLETE paragraph is stored as retrieval text.
    Highlighted answer spans are preserved in metadata.

MAUD:
    M&A agreement examples.
    Original question, answer, label, text type, and category
    information are preserved in metadata.

Run:
    python build_knowledge_base.py
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "data/knowledge_base"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "legal_records.jsonl",
)

# ------------------------------------------------------------
# LEDGAR
# ------------------------------------------------------------

# Maximum number of records retained from each LEDGAR category.
MAX_LEDGAR_PER_LABEL = 300

# ------------------------------------------------------------
# CUAD
# ------------------------------------------------------------

# Maximum number of CUAD QA records.
MAX_CUAD_RECORDS = 3000

# ------------------------------------------------------------
# MAUD
# ------------------------------------------------------------

# Maximum number of MAUD records.
MAX_MAUD_RECORDS = 3000

# ------------------------------------------------------------
# General
# ------------------------------------------------------------

MIN_TEXT_LENGTH = 80

# Correct Hugging Face repositories.
CUAD_REPOSITORY = "theatticusproject/cuad"
MAUD_REPOSITORY = "theatticusproject/maud"

# Actual CUAD file.
CUAD_FILENAME = "CUAD_v1/CUAD_v1.json"


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(value: Any) -> str:
    """
    Convert a value into clean single-line text.
    """

    if value is None:
        return ""

    text = str(value)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def valid_text(text: str) -> bool:
    """
    Reject extremely short text.
    """

    return len(text) >= MIN_TEXT_LENGTH


def clean_category(value: Any) -> str:
    """
    Clean category values.
    """

    category = clean_text(value)

    if not category:
        return "Unknown"

    return category


# ============================================================
# CUAD CATEGORY EXTRACTION
# ============================================================

def extract_category_from_question(
    question: str,
) -> str:
    """
    Extract the CUAD clause category from the question.

    Example:

        Highlight the parts (if any) of this contract
        related to "Termination" that should be reviewed
        by a lawyer.

    Returns:

        Termination
    """

    question = clean_text(question)

    if not question:
        return "Unknown"

    # --------------------------------------------------------
    # Normal CUAD format:
    #
    # related to "Termination"
    # --------------------------------------------------------

    quoted = re.search(
        r'"([^"]+)"',
        question,
    )

    if quoted:

        category = clean_text(
            quoted.group(1)
        )

        if category:
            return category

    # --------------------------------------------------------
    # Some versions may contain single quotes.
    # --------------------------------------------------------

    quoted_single = re.search(
        r"'([^']+)'",
        question,
    )

    if quoted_single:

        category = clean_text(
            quoted_single.group(1)
        )

        if category:
            return category

    return "Unknown"


# ============================================================
# GENERIC RECORD WRITER
# ============================================================

def write_record(
    file,
    source: str,
    source_id: str,
    title: str,
    category: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Write one normalized KB record.
    """

    record = {
        "source": source,
        "source_id": source_id,
        "title": title,
        "category": category,
        "text": text,
        "metadata": (
            metadata
            if metadata is not None
            else {}
        ),
    }

    file.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )


# ============================================================
# LEDGAR
# ============================================================

def add_ledgar_records(
    file,
) -> int:

    print("\nLoading LEDGAR...")

    dataset = load_dataset(
        "lex_glue",
        "ledgar",
        split="train",
    )

    label_names = (
        dataset
        .features["label"]
        .names
    )

    counts = Counter()

    total = 0

    for index, example in enumerate(dataset):

        label_id = example["label"]

        label_name = label_names[label_id]

        # Keep the KB reasonably balanced.
        if (
            counts[label_name]
            >= MAX_LEDGAR_PER_LABEL
        ):
            continue

        text = clean_text(
            example["text"]
        )

        if not valid_text(text):
            continue

        write_record(
            file=file,
            source="LEDGAR",
            source_id=f"ledgar_{index}",
            title=(
                f"LEDGAR — {label_name}"
            ),
            category=label_name,
            text=text,
            metadata={
                "original_label": label_name,
                "split": "train",
            },
        )

        counts[label_name] += 1
        total += 1

    print(
        f"LEDGAR records added: {total}"
    )

    print(
        f"LEDGAR categories used: "
        f"{len(counts)}"
    )

    return total


# ============================================================
# CUAD
# ============================================================

def add_cuad_records(
    file,
) -> int:

    print("\nLoading CUAD...")

    try:

        cuad_path = hf_hub_download(
            repo_id=CUAD_REPOSITORY,
            repo_type="dataset",
            filename=CUAD_FILENAME,
        )

        with open(
            cuad_path,
            "r",
            encoding="utf-8",
        ) as cuad_file:

            cuad_data = json.load(
                cuad_file
            )

    except Exception as error:

        print(
            "\nCUAD could not be "
            "downloaded or loaded."
        )

        print(
            f"Reason: "
            f"{error.__class__.__name__}: "
            f"{error}"
        )

        return 0

    total = 0
    category_counts = Counter()

    documents = cuad_data.get(
        "data",
        [],
    )

    # --------------------------------------------------------
    # CUAD structure:
    #
    # document
    #   └── paragraphs
    #          ├── context
    #          └── qas
    #                 ├── question
    #                 └── answers
    #
    # We create ONE KB record per QA.
    #
    # text:
    #     COMPLETE paragraph context
    #
    # metadata:
    #     question
    #     answer_texts
    #     category
    #     contract title
    #
    # This means retrieval gets the actual surrounding
    # contractual language instead of only the answer fragment.
    # --------------------------------------------------------

    for document_index, document in enumerate(
        documents
    ):

        if total >= MAX_CUAD_RECORDS:
            break

        contract_title = clean_text(
            document.get(
                "title",
                f"CUAD contract {document_index}",
            )
        )

        paragraphs = document.get(
            "paragraphs",
            [],
        )

        for paragraph_index, paragraph in enumerate(
            paragraphs
        ):

            if total >= MAX_CUAD_RECORDS:
                break

            context = clean_text(
                paragraph.get(
                    "context",
                    "",
                )
            )

            if not valid_text(context):
                continue

            questions = paragraph.get(
                "qas",
                [],
            )

            for question_index, qa in enumerate(
                questions
            ):

                if total >= MAX_CUAD_RECORDS:
                    break

                question = clean_text(
                    qa.get(
                        "question",
                        "",
                    )
                )

                category = (
                    extract_category_from_question(
                        question
                    )
                )

                answers = qa.get(
                    "answers",
                    [],
                )

                # ------------------------------------------------
                # Collect ALL answer spans.
                # ------------------------------------------------

                answer_texts = []

                for answer in answers:

                    answer_text = clean_text(
                        answer.get(
                            "text",
                            "",
                        )
                    )

                    if answer_text:
                        answer_texts.append(
                            answer_text
                        )

                # Remove duplicates while preserving order.
                answer_texts = list(
                    dict.fromkeys(
                        answer_texts
                    )
                )

                qa_id = clean_text(
                    qa.get(
                        "id",
                        "",
                    )
                )

                if not qa_id:

                    qa_id = (
                        f"cuad_"
                        f"{document_index}_"
                        f"{paragraph_index}_"
                        f"{question_index}"
                    )

                # ------------------------------------------------
                # ONE RECORD PER QA.
                #
                # Do NOT duplicate the same paragraph once for
                # every answer span.
                # ------------------------------------------------

                write_record(
                    file=file,
                    source="CUAD",
                    source_id=qa_id,
                    title=(
                        f"CUAD — {category}"
                    ),
                    category=category,
                    text=context,
                    metadata={
                        "contract_title": (
                            contract_title
                        ),

                        "question": (
                            question
                        ),

                        # The highlighted CUAD answers are
                        # preserved for reference/evaluation,
                        # NOT used as the retrieval text.
                        "answer_texts": (
                            answer_texts
                        ),

                        "paragraph_index": (
                            paragraph_index
                        ),

                        "question_index": (
                            question_index
                        ),

                        "qa_id": (
                            qa_id
                        ),

                        "dataset_category": (
                            category
                        ),

                        "has_answer": (
                            len(answer_texts) > 0
                        ),
                    },
                )

                category_counts[
                    category
                ] += 1

                total += 1

    print(
        f"CUAD records added: {total}"
    )

    print(
        f"CUAD categories used: "
        f"{len(category_counts)}"
    )

    if category_counts:

        print(
            "CUAD category examples:"
        )

        for category, count in (
            category_counts.most_common(10)
        ):

            print(
                f"  {category}: {count}"
            )

    return total


# ============================================================
# MAUD
# ============================================================

def add_maud_records(
    file,
) -> int:

    print("\nLoading MAUD...")

    try:

        dataset = load_dataset(
            MAUD_REPOSITORY,
        )

    except Exception as error:

        print(
            "\nMAUD could not be "
            "loaded."
        )

        print(
            f"Reason: "
            f"{error.__class__.__name__}: "
            f"{error}"
        )

        return 0

    total = 0
    category_counts = Counter()

    # --------------------------------------------------------
    # MAUD is retained because it gives the project an
    # additional M&A/legal transaction dimension.
    #
    # We DO NOT force MAUD into the same label structure as
    # LEDGAR or CUAD.
    #
    # Original fields are preserved in metadata.
    # --------------------------------------------------------

    for split_name in dataset.keys():

        if total >= MAX_MAUD_RECORDS:
            break

        print(
            f"Processing MAUD split: "
            f"{split_name}"
        )

        split = dataset[
            split_name
        ]

        for index, example in enumerate(
            split
        ):

            if total >= MAX_MAUD_RECORDS:
                break

            # ----------------------------------------------------
            # Extract available fields.
            # ----------------------------------------------------

            text = clean_text(
                example.get(
                    "text",
                    "",
                )
            )

            question = clean_text(
                example.get(
                    "question",
                    "",
                )
            )

            answer = clean_text(
                example.get(
                    "answer",
                    "",
                )
            )

            subquestion = clean_text(
                example.get(
                    "subquestion",
                    "",
                )
            )

            contract_name = clean_text(
                example.get(
                    "contract_name",
                    "",
                )
            )

            category = clean_category(
                example.get(
                    "category",
                    "",
                )
            )

            original_label = example.get(
                "label"
            )

            # ----------------------------------------------------
            # Try context if text is unavailable.
            # ----------------------------------------------------

            if not text:

                text = clean_text(
                    example.get(
                        "context",
                        "",
                    )
                )

            # ----------------------------------------------------
            # DO NOT normally use the answer as the main
            # retrieval text.
            #
            # Only use it as fallback if no context/text exists
            # and the answer itself is substantial.
            # ----------------------------------------------------

            if not text:

                if valid_text(answer):
                    text = answer

            if not valid_text(text):
                continue

            # ----------------------------------------------------
            # Record ID.
            # ----------------------------------------------------

            record_id = clean_text(
                example.get(
                    "id",
                    "",
                )
            )

            if not record_id:

                record_id = (
                    f"maud_"
                    f"{split_name}_"
                    f"{index}"
                )

            # ----------------------------------------------------
            # Preserve the original label.
            # ----------------------------------------------------

            label_text = clean_text(
                original_label
            )

            # If no category exists but a label does,
            # use the label only as a fallback grouping field.
            if (
                category == "Unknown"
                and label_text
            ):

                category = label_text

            # ----------------------------------------------------
            # Build title.
            # ----------------------------------------------------

            title_parts = [
                "MAUD",
                category,
            ]

            if question:

                title_parts.append(
                    question
                )

            title = " — ".join(
                title_parts
            )

            # ----------------------------------------------------
            # Write MAUD record.
            # ----------------------------------------------------

            write_record(
                file=file,
                source="MAUD",
                source_id=(
                    f"{record_id}_"
                    f"{split_name}_"
                    f"{index}"
                ),
                title=title,
                category=category,
                text=text,
                metadata={

                    "contract_name": (
                        contract_name
                    ),

                    "question": (
                        question
                    ),

                    "answer": (
                        answer
                    ),

                    "subquestion": (
                        subquestion
                    ),

                    "text_type": (
                        clean_text(
                            example.get(
                                "text_type",
                                "",
                            )
                        )
                    ),

                    "data_type": (
                        clean_text(
                            example.get(
                                "data_type",
                                "",
                            )
                        )
                    ),

                    # Original MAUD label.
                    "label": (
                        original_label
                    ),

                    "label_text": (
                        label_text
                    ),

                    "dataset_category": (
                        category
                    ),

                    "split": (
                        split_name
                    ),
                },
            )

            category_counts[
                category
            ] += 1

            total += 1

    print(
        f"MAUD records added: {total}"
    )

    print(
        f"MAUD categories used: "
        f"{len(category_counts)}"
    )

    if category_counts:

        print(
            "MAUD category examples:"
        )

        for category, count in (
            category_counts.most_common(10)
        ):

            print(
                f"  {category}: {count}"
            )

    return total


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print(
        "Building the legal "
        "knowledge base..."
    )

    print(
        f"Output file: "
        f"{OUTPUT_FILE}"
    )

    totals = {}

    # --------------------------------------------------------
    # "w" is intentional.
    #
    # Every execution completely rebuilds the KB so that old
    # records from the previous implementation do not remain.
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        # ====================================================
        # LEDGAR
        # ====================================================

        totals["LEDGAR"] = (
            add_ledgar_records(
                file
            )
        )

        # ====================================================
        # CUAD
        # ====================================================

        totals["CUAD"] = (
            add_cuad_records(
                file
            )
        )

        # ====================================================
        # MAUD
        # ====================================================

        totals["MAUD"] = (
            add_maud_records(
                file
            )
        )

    total_records = sum(
        totals.values()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "KNOWLEDGE BASE BUILD "
        "COMPLETED"
    )

    print(
        "=" * 60
    )

    for source, count in (
        totals.items()
    ):

        print(
            f"{source}: "
            f"{count} records"
        )

    print(
        f"\nTotal records: "
        f"{total_records}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Dataset warnings.
    # --------------------------------------------------------

    failed_sources = [
        source
        for source, count in totals.items()
        if count == 0
    ]

    if failed_sources:

        print(
            "\nWARNING:"
        )

        print(
            "The following datasets "
            "contributed zero records:"
        )

        for source in failed_sources:

            print(
                f"  - {source}"
            )

        print(
            "\nDo not continue to retrieval "
            "testing until the problem is "
            "resolved."
        )

    else:

        print(
            "\nAll three datasets "
            "contributed records successfully."
        )


if __name__ == "__main__":
    main()



# """
# Build a unified legal knowledge base from LEDGAR, CUAD, and MAUD.

# Output:

#     data/knowledge_base/legal_records.jsonl

# Each output record contains:

#     source
#     source_id
#     title
#     category
#     text
#     metadata

# The datasets are formatted differently, so each one is processed
# using dataset-specific extraction logic.

# Run:

#     python build_knowledge_base.py
# """

# from __future__ import annotations

# import json
# import os
# import re
# from collections import Counter
# from typing import Any

# from datasets import load_dataset
# from huggingface_hub import hf_hub_download


# # ============================================================
# # CONFIGURATION
# # ============================================================

# OUTPUT_DIR = "data/knowledge_base"

# OUTPUT_FILE = os.path.join(
#     OUTPUT_DIR,
#     "legal_records.jsonl",
# )

# # Keep LEDGAR balanced across its original categories.
# MAX_LEDGAR_PER_LABEL = 300

# # Maximum number of records retained from CUAD and MAUD.
# MAX_CUAD_RECORDS = 3000
# MAX_MAUD_RECORDS = 3000

# # Ignore extremely short legal text.
# MIN_TEXT_LENGTH = 80

# # Correct Hugging Face dataset repositories.
# CUAD_REPOSITORY = "theatticusproject/cuad"
# MAUD_REPOSITORY = "theatticusproject/maud"

# # Actual CUAD file inside the repository.
# CUAD_FILENAME = "CUAD_v1/CUAD_v1.json"


# # ============================================================
# # TEXT HELPERS
# # ============================================================

# def clean_text(value: Any) -> str:

#     if value is None:
#         return ""

#     text = str(value)

#     text = re.sub(
#         r"\s+",
#         " ",
#         text,
#     )

#     return text.strip()


# def valid_text(text: str) -> bool:

#     return len(text) >= MIN_TEXT_LENGTH


# def clean_category(value: Any) -> str:

#     category = clean_text(value)

#     if not category:
#         return "Unknown"

#     return category


# # ============================================================
# # CUAD CATEGORY EXTRACTION
# # ============================================================

# def extract_category_from_question(
#     question: str,
# ) -> str:
#     """
#     Extract the CUAD clause/question category.

#     CUAD uses questions such as:

#         Highlight the parts (if any) of this contract
#         related to "Termination" that should be reviewed
#         by a lawyer.

#     The text inside quotation marks is treated as the
#     clause category.

#     IMPORTANT:
#     The original question is also preserved in metadata.
#     """

#     question = clean_text(question)

#     if not question:
#         return "Unknown"

#     # Normal CUAD format.
#     quoted = re.search(
#         r'"([^"]+)"',
#         question,
#     )

#     if quoted:

#         category = clean_text(
#             quoted.group(1)
#         )

#         if category:
#             return category

#     # Some versions may contain single quotes.
#     quoted_single = re.search(
#         r"'([^']+)'",
#         question,
#     )

#     if quoted_single:

#         category = clean_text(
#             quoted_single.group(1)
#         )

#         if category:
#             return category

#     # If no category can be extracted, do NOT use the
#     # entire question as a fake category.
#     return "Unknown"


# # ============================================================
# # GENERIC RECORD WRITER
# # ============================================================

# def write_record(
#     file,
#     source: str,
#     source_id: str,
#     title: str,
#     category: str,
#     text: str,
#     metadata: dict[str, Any] | None = None,
# ) -> None:

#     record = {
#         "source": source,
#         "source_id": source_id,
#         "title": title,
#         "category": category,
#         "text": text,
#         "metadata": (
#             metadata
#             if metadata is not None
#             else {}
#         ),
#     }

#     file.write(
#         json.dumps(
#             record,
#             ensure_ascii=False,
#         )
#         + "\n"
#     )


# # ============================================================
# # LEDGAR
# # ============================================================

# def add_ledgar_records(
#     file,
# ) -> int:

#     print(
#         "\nLoading LEDGAR..."
#     )

#     dataset = load_dataset(
#         "lex_glue",
#         "ledgar",
#         split="train",
#     )

#     label_names = (
#         dataset
#         .features["label"]
#         .names
#     )

#     counts = Counter()

#     total = 0

#     for index, example in enumerate(
#         dataset
#     ):

#         label_id = example["label"]

#         label_name = (
#             label_names[label_id]
#         )

#         if (
#             counts[label_name]
#             >= MAX_LEDGAR_PER_LABEL
#         ):
#             continue

#         text = clean_text(
#             example["text"]
#         )

#         if not valid_text(text):
#             continue

#         write_record(
#             file=file,
#             source="LEDGAR",
#             source_id=f"ledgar_{index}",
#             title=(
#                 f"LEDGAR — "
#                 f"{label_name}"
#             ),
#             category=label_name,
#             text=text,
#             metadata={
#                 "original_label": label_name,
#                 "split": "train",
#             },
#         )

#         counts[label_name] += 1
#         total += 1

#     print(
#         f"LEDGAR records added: {total}"
#     )

#     print(
#         f"LEDGAR categories used: "
#         f"{len(counts)}"
#     )

#     return total


# # ============================================================
# # CUAD
# # ============================================================

# def add_cuad_records(
#     file,
# ) -> int:

#     print(
#         "\nLoading CUAD..."
#     )

#     try:

#         cuad_path = hf_hub_download(
#             repo_id=CUAD_REPOSITORY,
#             repo_type="dataset",
#             filename=CUAD_FILENAME,
#         )

#         with open(
#             cuad_path,
#             "r",
#             encoding="utf-8",
#         ) as cuad_file:

#             cuad_data = json.load(
#                 cuad_file
#             )

#     except Exception as error:

#         print(
#             "\nCUAD could not be "
#             "downloaded or loaded."
#         )

#         print(
#             f"Reason: "
#             f"{error.__class__.__name__}: "
#             f"{error}"
#         )

#         return 0

#     total = 0
#     category_counts = Counter()

#     documents = cuad_data.get(
#         "data",
#         [],
#     )

#     # --------------------------------------------------------
#     # CUAD is SQuAD-style:
#     #
#     # document
#     #   └── paragraphs
#     #          ├── context
#     #          └── qas
#     #                 ├── question
#     #                 └── answers
#     #
#     # We convert each QA into a KB record.
#     #
#     # IMPORTANT CHANGE:
#     #
#     # The COMPLETE paragraph context is stored as "text".
#     #
#     # The highlighted CUAD answer is stored separately in
#     # metadata["answer_text"].
#     #
#     # This gives retrieval the full legal clause/context
#     # rather than only the highlighted answer fragment.
#     # --------------------------------------------------------

#     for document_index, document in enumerate(
#         documents
#     ):

#         if total >= MAX_CUAD_RECORDS:
#             break

#         contract_title = clean_text(
#             document.get(
#                 "title",
#                 f"CUAD contract {document_index}",
#             )
#         )

#         paragraphs = document.get(
#             "paragraphs",
#             [],
#         )

#         for paragraph_index, paragraph in enumerate(
#             paragraphs
#         ):

#             if total >= MAX_CUAD_RECORDS:
#                 break

#             context = clean_text(
#                 paragraph.get(
#                     "context",
#                     "",
#                 )
#             )

#             if not valid_text(context):
#                 continue

#             questions = paragraph.get(
#                 "qas",
#                 [],
#             )

#             for question_index, qa in enumerate(
#                 questions
#             ):

#                 if total >= MAX_CUAD_RECORDS:
#                     break

#                 question = clean_text(
#                     qa.get(
#                         "question",
#                         "",
#                     )
#                 )

#                 category = (
#                     extract_category_from_question(
#                         question
#                     )
#                 )

#                 answers = qa.get(
#                     "answers",
#                     [],
#                 )

#                 # ------------------------------------------------
#                 # Collect the answer spans.
#                 # ------------------------------------------------

#                 answer_texts = []

#                 for answer in answers:

#                     answer_text = clean_text(
#                         answer.get(
#                             "text",
#                             "",
#                         )
#                     )

#                     if answer_text:
#                         answer_texts.append(
#                             answer_text
#                         )

#                 # Remove duplicates while preserving order.
#                 unique_answers = list(
#                     dict.fromkeys(
#                         answer_texts
#                     )
#                 )

#                 # ------------------------------------------------
#                 # If the QA has no answer span, still retain the
#                 # legal context. This is important because some
#                 # CUAD questions represent "no answer" cases.
#                 # ------------------------------------------------

#                 if not unique_answers:

#                     unique_answers = [""]

#                 qa_id = clean_text(
#                     qa.get(
#                         "id",
#                         "",
#                     )
#                 )

#                 if not qa_id:

#                     qa_id = (
#                         f"cuad_"
#                         f"{document_index}_"
#                         f"{paragraph_index}_"
#                         f"{question_index}"
#                     )

#                 # ------------------------------------------------
#                 # Create one KB record for each unique answer.
#                 #
#                 # The RETRIEVAL TEXT is ALWAYS the complete
#                 # paragraph context.
#                 #
#                 # The answer span is metadata.
#                 # ------------------------------------------------

#                 for answer_number, answer_text in enumerate(
#                     unique_answers
#                 ):

#                     if total >= MAX_CUAD_RECORDS:
#                         break

#                     record_id = (
#                         f"{qa_id}_"
#                         f"{answer_number}"
#                     )

#                     write_record(
#                         file=file,
#                         source="CUAD",
#                         source_id=record_id,
#                         title=(
#                             f"CUAD — "
#                             f"{category}"
#                         ),
#                         category=category,
#                         text=context,
#                         metadata={
#                             "contract_title": (
#                                 contract_title
#                             ),

#                             # Preserve the original
#                             # classification-style question.
#                             "question": (
#                                 question
#                             ),

#                             # Preserve the actual highlighted
#                             # answer span from CUAD.
#                             "answer_text": (
#                                 answer_text
#                             ),

#                             "paragraph_index": (
#                                 paragraph_index
#                             ),

#                             "question_index": (
#                                 question_index
#                             ),

#                             "answer_index": (
#                                 answer_number
#                             ),

#                             "qa_id": (
#                                 qa_id
#                             ),

#                             "dataset_category": (
#                                 category
#                             ),

#                             "has_answer": (
#                                 bool(answer_text)
#                             ),
#                         },
#                     )

#                     category_counts[
#                         category
#                     ] += 1

#                     total += 1

#     print(
#         f"CUAD records added: {total}"
#     )

#     print(
#         f"CUAD categories used: "
#         f"{len(category_counts)}"
#     )

#     if category_counts:

#         print(
#             "CUAD category examples:"
#         )

#         for category, count in (
#             category_counts.most_common(10)
#         ):

#             print(
#                 f"  {category}: {count}"
#             )

#     return total


# # ============================================================
# # MAUD
# # ============================================================

# def add_maud_records(
#     file,
# ) -> int:

#     print(
#         "\nLoading MAUD..."
#     )

#     try:

#         dataset = load_dataset(
#             MAUD_REPOSITORY,
#         )

#     except Exception as error:

#         print(
#             "\nMAUD could not be "
#             "loaded."
#         )

#         print(
#             f"Reason: "
#             f"{error.__class__.__name__}: "
#             f"{error}"
#         )

#         return 0

#     total = 0
#     category_counts = Counter()

#     # --------------------------------------------------------
#     # IMPORTANT:
#     #
#     # Do not assume that MAUD's "category" field is necessarily
#     # the same type of label as LEDGAR.
#     #
#     # We preserve:
#     #
#     #   category
#     #   label
#     #   question
#     #   subquestion
#     #   answer
#     #
#     # inside the record metadata.
#     #
#     # The original values are therefore not lost.
#     # --------------------------------------------------------

#     for split_name in dataset.keys():

#         if total >= MAX_MAUD_RECORDS:
#             break

#         print(
#             f"Processing MAUD split: "
#             f"{split_name}"
#         )

#         split = dataset[
#             split_name
#         ]

#         for index, example in enumerate(
#             split
#         ):

#             if total >= MAX_MAUD_RECORDS:
#                 break

#             # ----------------------------------------------------
#             # MAUD may use different field structures depending
#             # on the dataset version.
#             #
#             # Safely inspect the fields we actually receive.
#             # ----------------------------------------------------

#             text = clean_text(
#                 example.get(
#                     "text",
#                     "",
#                 )
#             )

#             question = clean_text(
#                 example.get(
#                     "question",
#                     "",
#                 )
#             )

#             answer = clean_text(
#                 example.get(
#                     "answer",
#                     "",
#                 )
#             )

#             subquestion = clean_text(
#                 example.get(
#                     "subquestion",
#                     "",
#                 )
#             )

#             contract_name = clean_text(
#                 example.get(
#                     "contract_name",
#                     "",
#                 )
#             )

#             category = clean_category(
#                 example.get(
#                     "category",
#                     "",
#                 )
#             )

#             original_label = example.get(
#                 "label"
#             )

#             # ----------------------------------------------------
#             # If the dataset does not provide "text", try other
#             # text-bearing fields before discarding the record.
#             # ----------------------------------------------------

#             if not text:

#                 # Some versions may use "context".
#                 text = clean_text(
#                     example.get(
#                         "context",
#                         "",
#                     )
#                 )

#             if not text:

#                 # If context is unavailable, use the answer
#                 # only when it is sufficiently meaningful.
#                 if valid_text(answer):
#                     text = answer

#             if not valid_text(text):
#                 continue

#             record_id = clean_text(
#                 example.get(
#                     "id",
#                     "",
#                 )
#             )

#             if not record_id:

#                 record_id = (
#                     f"maud_"
#                     f"{split_name}_"
#                     f"{index}"
#                 )

#             # ----------------------------------------------------
#             # Preserve the actual dataset label separately.
#             #
#             # This is important because "category" and "label"
#             # are not assumed to mean the same thing.
#             # ----------------------------------------------------

#             label_text = clean_text(
#                 original_label
#             )

#             # If category is missing but label exists, use the
#             # label as the category for retrieval grouping.
#             if (
#                 category == "Unknown"
#                 and label_text
#             ):

#                 category = label_text

#             title_parts = [
#                 "MAUD",
#                 category,
#             ]

#             if question:

#                 title_parts.append(
#                     question
#                 )

#             write_record(
#                 file=file,
#                 source="MAUD",
#                 source_id=(
#                     f"{record_id}_"
#                     f"{split_name}_"
#                     f"{index}"
#                 ),
#                 title=(
#                     " — ".join(
#                         title_parts
#                     )
#                 ),
#                 category=category,
#                 text=text,
#                 metadata={

#                     "contract_name": (
#                         contract_name
#                     ),

#                     "question": (
#                         question
#                     ),

#                     "answer": (
#                         answer
#                     ),

#                     "subquestion": (
#                         subquestion
#                     ),

#                     "text_type": (
#                         clean_text(
#                             example.get(
#                                 "text_type",
#                                 "",
#                             )
#                         )
#                     ),

#                     "data_type": (
#                         clean_text(
#                             example.get(
#                                 "data_type",
#                                 "",
#                             )
#                         )
#                     ),

#                     # Preserve original MAUD label exactly.
#                     "label": (
#                         original_label
#                     ),

#                     "label_text": (
#                         label_text
#                     ),

#                     "dataset_category": (
#                         category
#                     ),

#                     "split": (
#                         split_name
#                     ),
#                 },
#             )

#             category_counts[
#                 category
#             ] += 1

#             total += 1

#     print(
#         f"MAUD records added: {total}"
#     )

#     print(
#         f"MAUD categories used: "
#         f"{len(category_counts)}"
#     )

#     if category_counts:

#         print(
#             "MAUD category examples:"
#         )

#         for category, count in (
#             category_counts.most_common(10)
#         ):

#             print(
#                 f"  {category}: {count}"
#             )

#     return total


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     os.makedirs(
#         OUTPUT_DIR,
#         exist_ok=True,
#     )

#     print(
#         "Building the legal "
#         "knowledge base..."
#     )

#     print(
#         f"Output file: "
#         f"{OUTPUT_FILE}"
#     )

#     totals = {}

#     with open(
#         OUTPUT_FILE,
#         "w",
#         encoding="utf-8",
#     ) as file:

#         totals["LEDGAR"] = (
#             add_ledgar_records(
#                 file
#             )
#         )

#         totals["CUAD"] = (
#             add_cuad_records(
#                 file
#             )
#         )

#         totals["MAUD"] = (
#             add_maud_records(
#                 file
#             )
#         )

#     total_records = sum(
#         totals.values()
#     )

#     print(
#         "\n"
#         + "=" * 60
#     )

#     print(
#         "KNOWLEDGE BASE BUILD "
#         "COMPLETED"
#     )

#     print(
#         "=" * 60
#     )

#     for source, count in (
#         totals.items()
#     ):

#         print(
#             f"{source}: "
#             f"{count} records"
#         )

#     print(
#         f"\nTotal records: "
#         f"{total_records}"
#     )

#     print(
#         f"Saved to: "
#         f"{OUTPUT_FILE}"
#     )

#     if (
#         totals["CUAD"] == 0
#         or totals["MAUD"] == 0
#     ):

#         print(
#             "\nWARNING:"
#         )

#         print(
#             "One or more datasets "
#             "contributed zero records."
#         )

#         print(
#             "Do not continue to the "
#             "retrieval implementation "
#             "until this is resolved."
#         )


# if __name__ == "__main__":

#     main()



# # """
# # Build a unified legal knowledge base from LEDGAR, CUAD, and MAUD.

# # Output:

# #     data/knowledge_base/legal_records.jsonl

# # Each output record contains:

# #     source
# #     source_id
# #     title
# #     category
# #     text
# #     metadata

# # The datasets are formatted differently, so each one is processed
# # using dataset-specific extraction logic.

# # Run:

# #     python build_knowledge_base.py
# # """

# # from __future__ import annotations

# # import json
# # import os
# # import re
# # from collections import Counter
# # from typing import Any

# # from datasets import load_dataset
# # from huggingface_hub import hf_hub_download


# # # ============================================================
# # # CONFIGURATION
# # # ============================================================

# # OUTPUT_DIR = "data/knowledge_base"

# # OUTPUT_FILE = os.path.join(
# #     OUTPUT_DIR,
# #     "legal_records.jsonl",
# # )

# # # Keep LEDGAR balanced across its original categories.
# # MAX_LEDGAR_PER_LABEL = 300

# # # Maximum number of records retained from CUAD and MAUD.
# # MAX_CUAD_RECORDS = 3000
# # MAX_MAUD_RECORDS = 3000

# # # Ignore very short text fragments.
# # MIN_TEXT_LENGTH = 80

# # # Correct Hugging Face dataset repositories.
# # CUAD_REPOSITORY = "theatticusproject/cuad"
# # MAUD_REPOSITORY = "theatticusproject/maud"

# # # Actual CUAD file inside the repository.
# # CUAD_FILENAME = "CUAD_v1/CUAD_v1.json"


# # # ============================================================
# # # TEXT HELPERS
# # # ============================================================

# # def clean_text(value: Any) -> str:

# #     if value is None:
# #         return ""

# #     text = str(value)

# #     text = re.sub(
# #         r"\s+",
# #         " ",
# #         text,
# #     )

# #     return text.strip()


# # def valid_text(text: str) -> bool:

# #     return len(text) >= MIN_TEXT_LENGTH


# # def clean_category(value: Any) -> str:

# #     category = clean_text(
# #         value
# #     )

# #     if not category:
# #         return "Unknown"

# #     return category


# # def extract_category_from_question(
# #     question: str,
# # ) -> str:
# #     """
# #     CUAD questions commonly look like:

# #     Highlight the parts (if any) of this contract related to
# #     "Termination" that should be reviewed by a lawyer.

# #     Extract the category inside quotation marks.
# #     """

# #     question = clean_text(
# #         question
# #     )

# #     quoted = re.search(
# #         r'"([^"]+)"',
# #         question,
# #     )

# #     if quoted:

# #         category = clean_text(
# #             quoted.group(1)
# #         )

# #         if category:

# #             return category

# #     # Fallback when the expected quoted format is absent.
# #     return question or "CUAD"


# # def write_record(
# #     file,
# #     source: str,
# #     source_id: str,
# #     title: str,
# #     category: str,
# #     text: str,
# #     metadata: dict[str, Any] | None = None,
# # ) -> None:

# #     record = {
# #         "source": source,
# #         "source_id": source_id,
# #         "title": title,
# #         "category": category,
# #         "text": text,
# #         "metadata": (
# #             metadata
# #             if metadata is not None
# #             else {}
# #         ),
# #     }

# #     file.write(
# #         json.dumps(
# #             record,
# #             ensure_ascii=False,
# #         )
# #         + "\n"
# #     )


# # # ============================================================
# # # LEDGAR
# # # ============================================================

# # def add_ledgar_records(
# #     file,
# # ) -> int:

# #     print(
# #         "\nLoading LEDGAR..."
# #     )

# #     dataset = load_dataset(
# #         "lex_glue",
# #         "ledgar",
# #         split="train",
# #     )

# #     label_names = (
# #         dataset
# #         .features["label"]
# #         .names
# #     )

# #     counts = Counter()

# #     total = 0

# #     for index, example in enumerate(
# #         dataset
# #     ):

# #         label_id = (
# #             example["label"]
# #         )

# #         label_name = (
# #             label_names[label_id]
# #         )

# #         if (
# #             counts[label_name]
# #             >= MAX_LEDGAR_PER_LABEL
# #         ):

# #             continue

# #         text = clean_text(
# #             example["text"]
# #         )

# #         if not valid_text(
# #             text
# #         ):

# #             continue

# #         write_record(
# #             file=file,
# #             source="LEDGAR",
# #             source_id=(
# #                 f"ledgar_{index}"
# #             ),
# #             title=(
# #                 f"LEDGAR — "
# #                 f"{label_name}"
# #             ),
# #             category=label_name,
# #             text=text,
# #             metadata={
# #                 "original_label": (
# #                     label_name
# #                 ),
# #                 "split": "train",
# #             },
# #         )

# #         counts[label_name] += 1

# #         total += 1

# #     print(
# #         f"LEDGAR records added: "
# #         f"{total}"
# #     )

# #     print(
# #         f"LEDGAR categories used: "
# #         f"{len(counts)}"
# #     )

# #     return total


# # # ============================================================
# # # CUAD
# # # ============================================================

# # def add_cuad_records(
# #     file,
# # ) -> int:

# #     print(
# #         "\nLoading CUAD..."
# #     )

# #     try:

# #         # Download the actual SQuAD-style CUAD JSON file.
# #         cuad_path = (
# #             hf_hub_download(
# #                 repo_id=(
# #                     CUAD_REPOSITORY
# #                 ),
# #                 repo_type="dataset",
# #                 filename=(
# #                     CUAD_FILENAME
# #                 ),
# #             )
# #         )

# #         with open(
# #             cuad_path,
# #             "r",
# #             encoding="utf-8",
# #         ) as cuad_file:

# #             cuad_data = json.load(
# #                 cuad_file
# #             )

# #     except Exception as error:

# #         print(
# #             "\nCUAD could not be "
# #             "downloaded or loaded."
# #         )

# #         print(
# #             f"Reason: "
# #             f"{error.__class__.__name__}: "
# #             f"{error}"
# #         )

# #         return 0

# #     total = 0

# #     category_counts = Counter()

# #     # CUAD uses the standard SQuAD-style:
# #     #
# #     # {
# #     #   "data": [
# #     #       {
# #     #           "title": "...",
# #     #           "paragraphs": [
# #     #               {
# #     #                   "context": "...",
# #     #                   "qas": [...]
# #     #               }
# #     #           ]
# #     #       }
# #     #   ]
# #     # }

# #     documents = (
# #         cuad_data.get(
# #             "data",
# #             [],
# #         )
# #     )

# #     for document_index, document in enumerate(
# #         documents
# #     ):

# #         if (
# #             total
# #             >= MAX_CUAD_RECORDS
# #         ):

# #             break

# #         contract_title = clean_text(
# #             document.get(
# #                 "title",
# #                 f"CUAD contract "
# #                 f"{document_index}",
# #             )
# #         )

# #         paragraphs = (
# #             document.get(
# #                 "paragraphs",
# #                 [],
# #             )
# #         )

# #         for paragraph_index, paragraph in enumerate(
# #             paragraphs
# #         ):

# #             if (
# #                 total
# #                 >= MAX_CUAD_RECORDS
# #             ):

# #                 break

# #             context = clean_text(
# #                 paragraph.get(
# #                     "context",
# #                     "",
# #                 )
# #             )

# #             if not valid_text(
# #                 context
# #             ):

# #                 continue

# #             questions = (
# #                 paragraph.get(
# #                     "qas",
# #                     [],
# #                 )
# #             )

# #             for question_index, qa in enumerate(
# #                 questions
# #             ):

# #                 if (
# #                     total
# #                     >= MAX_CUAD_RECORDS
# #                 ):

# #                     break

# #                 question = clean_text(
# #                     qa.get(
# #                         "question",
# #                         "",
# #                     )
# #                 )

# #                 category = (
# #                     extract_category_from_question(
# #                         question
# #                     )
# #                 )

# #                 answers = (
# #                     qa.get(
# #                         "answers",
# #                         [],
# #                     )
# #                 )

# #                 # CUAD may have multiple answer spans.
# #                 # Store each unique answer span as a legal
# #                 # knowledge record.
# #                 unique_answers = set()

# #                 for answer in answers:

# #                     answer_text = clean_text(
# #                         answer.get(
# #                             "text",
# #                             "",
# #                         )
# #                     )

# #                     if not answer_text:

# #                         continue

# #                     unique_answers.add(
# #                         answer_text
# #                     )

# #                 # If answer spans are too short, use the
# #                 # paragraph context because retrieval needs
# #                 # meaningful legal evidence.
# #                 if not unique_answers:

# #                     if valid_text(
# #                         context
# #                     ):

# #                         unique_answers.add(
# #                             context
# #                         )

# #                 for answer_number, answer_text in enumerate(
# #                     unique_answers
# #                 ):

# #                     # A short answer by itself may be something
# #                     # like a date or number. In that case, retain
# #                     # the complete legal context.
# #                     if not valid_text(
# #                         answer_text
# #                     ):

# #                         text = context

# #                     else:

# #                         text = answer_text

# #                     if not valid_text(
# #                         text
# #                     ):

# #                         continue

# #                     qa_id = clean_text(
# #                         qa.get(
# #                             "id",
# #                             "",
# #                         )
# #                     )

# #                     if not qa_id:

# #                         qa_id = (
# #                             f"cuad_"
# #                             f"{document_index}_"
# #                             f"{paragraph_index}_"
# #                             f"{question_index}_"
# #                             f"{answer_number}"
# #                         )

# #                     write_record(
# #                         file=file,
# #                         source="CUAD",
# #                         source_id=(
# #                             f"{qa_id}_"
# #                             f"{answer_number}"
# #                         ),
# #                         title=(
# #                             f"CUAD — "
# #                             f"{category}"
# #                         ),
# #                         category=category,
# #                         text=text,
# #                         metadata={
# #                             "contract_title": (
# #                                 contract_title
# #                             ),
# #                             "question": (
# #                                 question
# #                             ),
# #                             "paragraph_index": (
# #                                 paragraph_index
# #                             ),
# #                             "answer_text": (
# #                                 answer_text
# #                             ),
# #                         },
# #                     )

# #                     category_counts[
# #                         category
# #                     ] += 1

# #                     total += 1

# #                     if (
# #                         total
# #                         >= MAX_CUAD_RECORDS
# #                     ):

# #                         break

# #     print(
# #         f"CUAD records added: "
# #         f"{total}"
# #     )

# #     print(
# #         f"CUAD categories used: "
# #         f"{len(category_counts)}"
# #     )

# #     return total


# # # ============================================================
# # # MAUD
# # # ============================================================

# # def add_maud_records(
# #     file,
# # ) -> int:

# #     print(
# #         "\nLoading MAUD..."
# #     )

# #     try:

# #         dataset = load_dataset(
# #             MAUD_REPOSITORY,
# #         )

# #     except Exception as error:

# #         print(
# #             "\nMAUD could not be "
# #             "loaded."
# #         )

# #         print(
# #             f"Reason: "
# #             f"{error.__class__.__name__}: "
# #             f"{error}"
# #         )

# #         return 0

# #     total = 0

# #     category_counts = Counter()

# #     # MAUD has multiple splits. Process all available
# #     # splits until the configured maximum is reached.
# #     for split_name in dataset.keys():

# #         if (
# #             total
# #             >= MAX_MAUD_RECORDS
# #         ):

# #             break

# #         print(
# #             f"Processing MAUD split: "
# #             f"{split_name}"
# #         )

# #         split = (
# #             dataset[
# #                 split_name
# #             ]
# #         )

# #         for index, example in enumerate(
# #             split
# #         ):

# #             if (
# #                 total
# #                 >= MAX_MAUD_RECORDS
# #             ):

# #                 break

# #             text = clean_text(
# #                 example.get(
# #                     "text",
# #                     "",
# #                 )
# #             )

# #             if not valid_text(
# #                 text
# #             ):

# #                 continue

# #             category = clean_category(
# #                 example.get(
# #                     "category",
# #                     ""
# #                 )
# #             )

# #             question = clean_text(
# #                 example.get(
# #                     "question",
# #                     ""
# #                 )
# #             )

# #             answer = clean_text(
# #                 example.get(
# #                     "answer",
# #                     ""
# #                 )
# #             )

# #             contract_name = clean_text(
# #                 example.get(
# #                     "contract_name",
# #                     ""
# #                 )
# #             )

# #             record_id = clean_text(
# #                 example.get(
# #                     "id",
# #                     ""
# #                 )
# #             )

# #             if not record_id:

# #                 record_id = (
# #                     f"maud_"
# #                     f"{split_name}_"
# #                     f"{index}"
# #                 )

# #             title_parts = [
# #                 "MAUD",
# #                 category,
# #             ]

# #             if question:

# #                 title_parts.append(
# #                     question
# #                 )

# #             write_record(
# #                 file=file,
# #                 source="MAUD",
# #                 source_id=(
# #                     f"{record_id}_"
# #                     f"{split_name}_"
# #                     f"{index}"
# #                 ),
# #                 title=(
# #                     " — ".join(
# #                         title_parts
# #                     )
# #                 ),
# #                 category=category,
# #                 text=text,
# #                 metadata={
# #                     "contract_name": (
# #                         contract_name
# #                     ),
# #                     "question": (
# #                         question
# #                     ),
# #                     "answer": (
# #                         answer
# #                     ),
# #                     "subquestion": (
# #                         clean_text(
# #                             example.get(
# #                                 "subquestion",
# #                                 "",
# #                             )
# #                         )
# #                     ),
# #                     "text_type": (
# #                         clean_text(
# #                             example.get(
# #                                 "text_type",
# #                                 "",
# #                             )
# #                         )
# #                     ),
# #                     "data_type": (
# #                         clean_text(
# #                             example.get(
# #                                 "data_type",
# #                                 "",
# #                             )
# #                         )
# #                     ),
# #                     "label": (
# #                         example.get(
# #                             "label"
# #                         )
# #                     ),
# #                     "split": (
# #                         split_name
# #                     ),
# #                 },
# #             )

# #             category_counts[
# #                 category
# #             ] += 1

# #             total += 1

# #     print(
# #         f"MAUD records added: "
# #         f"{total}"
# #     )

# #     print(
# #         f"MAUD categories used: "
# #         f"{len(category_counts)}"
# #     )

# #     return total


# # # ============================================================
# # # MAIN
# # # ============================================================

# # def main():

# #     os.makedirs(
# #         OUTPUT_DIR,
# #         exist_ok=True,
# #     )

# #     print(
# #         "Building the legal "
# #         "knowledge base..."
# #     )

# #     print(
# #         f"Output file: "
# #         f"{OUTPUT_FILE}"
# #     )

# #     totals = {}

# #     with open(
# #         OUTPUT_FILE,
# #         "w",
# #         encoding="utf-8",
# #     ) as file:

# #         totals["LEDGAR"] = (
# #             add_ledgar_records(
# #                 file
# #             )
# #         )

# #         totals["CUAD"] = (
# #             add_cuad_records(
# #                 file
# #             )
# #         )

# #         totals["MAUD"] = (
# #             add_maud_records(
# #                 file
# #             )
# #         )

# #     total_records = sum(
# #         totals.values()
# #     )

# #     print(
# #         "\n"
# #         + "=" * 60
# #     )

# #     print(
# #         "KNOWLEDGE BASE BUILD "
# #         "COMPLETED"
# #     )

# #     print(
# #         "=" * 60
# #     )

# #     for source, count in (
# #         totals.items()
# #     ):

# #         print(
# #             f"{source}: "
# #             f"{count} records"
# #         )

# #     print(
# #         f"\nTotal records: "
# #         f"{total_records}"
# #     )

# #     print(
# #         f"Saved to: "
# #         f"{OUTPUT_FILE}"
# #     )

# #     if (
# #         totals["CUAD"] == 0
# #         or totals["MAUD"] == 0
# #     ):

# #         print(
# #             "\nWARNING:"
# #         )

# #         print(
# #             "One or more datasets "
# #             "contributed zero records."
# #         )

# #         print(
# #             "Do not continue to the "
# #             "retrieval implementation "
# #             "until this is resolved."
# #         )


# # if __name__ == "__main__":

# #     main()