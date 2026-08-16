"""
Build a persistent FAISS index for the unified legal knowledge base.

Input:
    data/knowledge_base/legal_records.jsonl

Output:
    data/knowledge_base/legal_faiss.index
    data/knowledge_base/legal_faiss_metadata.json

Run:

    python build_faiss_index.py

IMPORTANT:
    This script needs to be run whenever legal_records.jsonl changes.

    After the index is built, normal contract analysis does NOT
    regenerate these 33,328 embeddings.
"""

from __future__ import annotations

import json
import os

import faiss
import numpy as np

from embeddings import get_encoder


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

KB_DIR = os.path.join(
    BASE_DIR,
    "data",
    "knowledge_base",
)

RECORDS_FILE = os.path.join(
    KB_DIR,
    "legal_records.jsonl",
)

INDEX_FILE = os.path.join(
    KB_DIR,
    "legal_faiss.index",
)

METADATA_FILE = os.path.join(
    KB_DIR,
    "legal_faiss_metadata.json",
)


# ============================================================
# SETTINGS
# ============================================================

# Number of records processed by LegalBERT at once.
#
# 64 is reasonable if the machine is stable.
# Reduce to 32 if GPU/RAM becomes a problem.
BATCH_SIZE = 64


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_records() -> list[dict]:

    if not os.path.isfile(
        RECORDS_FILE
    ):

        raise FileNotFoundError(

            "\nKnowledge-base file was not found.\n\n"

            f"Expected:\n"
            f"{RECORDS_FILE}\n\n"

            "Run:\n\n"

            "    python build_knowledge_base.py\n"

        )

    records = []

    with open(
        RECORDS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as error:

                print(
                    "[FAISS] "
                    f"Skipping invalid JSON on line "
                    f"{line_number}: {error}"
                )

    if not records:

        raise RuntimeError(
            "The knowledge-base contains no records."
        )

    return records


# ============================================================
# RECORD -> EMBEDDING TEXT
# ============================================================

def record_to_text(
    record: dict,
) -> str:

    category = str(
        record.get(
            "category",
            "",
        )
    )

    title = str(
        record.get(
            "title",
            "",
        )
    )

    text = str(
        record.get(
            "text",
            "",
        )
    )

    return (

        f"Category: {category}. "

        f"Title: {title}. "

        f"Clause: {text}"

    )


# ============================================================
# BUILD INDEX
# ============================================================

def build_index():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BUILDING LEGAL FAISS INDEX"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Load records
    # --------------------------------------------------------

    records = load_records()

    total = len(
        records
    )

    print(
        f"\nRecords loaded: {total:,}"
    )


    # --------------------------------------------------------
    # Load shared LegalBERT encoder
    # --------------------------------------------------------

    print(
        "\nLoading LegalBERT embedding encoder..."
    )

    encoder = get_encoder()


    print(
        f"Embedding backend: {encoder.name}"
    )


    # --------------------------------------------------------
    # Generate embeddings in batches
    # --------------------------------------------------------

    print(
        "\nGenerating embeddings..."
    )

    embeddings = []

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = records[
            start:end
        ]

        texts = [

            record_to_text(
                record
            )

            for record
            in batch

        ]


        vectors = encoder.encode_batch(

            texts,

            batch_size=BATCH_SIZE,

            show_progress=False,

        )


        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )


        embeddings.append(
            vectors
        )


        print(

            f"\rEmbedded "
            f"{end:,}/{total:,} "
            f"({end / total * 100:.1f}%)",

            end="",

            flush=True,

        )


    print()


    # --------------------------------------------------------
    # Combine embeddings
    # --------------------------------------------------------

    matrix = np.vstack(
        embeddings
    ).astype(
        np.float32
    )


    if matrix.shape[0] != total:

        raise RuntimeError(

            "Embedding count does not match "
            "knowledge-base record count."

        )


    print(
        "\nEmbedding matrix:",
        matrix.shape
    )


    # --------------------------------------------------------
    # Normalize
    #
    # Inner product of normalized vectors =
    # cosine similarity.
    # --------------------------------------------------------

    faiss.normalize_L2(
        matrix
    )


    dimension = (
        matrix.shape[1]
    )


    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    print(
        "\nCreating FAISS IndexFlatIP..."
    )


    index = faiss.IndexFlatIP(
        dimension
    )


    index.add(
        matrix
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        KB_DIR,
        exist_ok=True,
    )


    print(
        "Saving FAISS index..."
    )

    faiss.write_index(
        index,
        INDEX_FILE,
    )


    print(
        "Saving metadata..."
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
        )


    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FAISS INDEX BUILD COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nIndexed records: "
        f"{index.ntotal:,}"
    )

    print(
        f"Embedding dimension: "
        f"{dimension}"
    )

    print(
        f"\nIndex:"
        f"\n{INDEX_FILE}"
    )

    print(
        f"\nMetadata:"
        f"\n{METADATA_FILE}"
    )

    print(
        "\nYou do NOT need to run this again "
        "for every contract."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_index()