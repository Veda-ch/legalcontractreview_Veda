# from __future__ import annotations

# import json
# import os
# import re
# from collections import defaultdict

# import networkx as nx
# import numpy as np

# from embeddings import get_encoder, cosine_similarity
# from interfaces import RetrievalHit, RetrievalResult


# # ============================================================
# # CONFIGURATION
# # ============================================================

# BASE_DIR = os.path.dirname(
#     os.path.abspath(__file__)
# )

# KNOWLEDGE_BASE_FILE = os.path.join(
#     BASE_DIR,
#     "data",
#     "knowledge_base",
#     "legal_records.jsonl",
# )

# # Maximum number of records sent to semantic scoring.
# #
# # This keeps retrieval fast and avoids embedding the entire
# # knowledge base.
# MAX_GRAPH_CANDIDATES = 120

# # Maximum number of records retained from one category.
# MAX_RECORDS_PER_CATEGORY = 40

# # Maximum number of candidates taken from one source while
# # building the graph candidate pool.
# #
# # IMPORTANT:
# # This does NOT force every source into the final results.
# # It only prevents one source from filling the entire
# # candidate pool before semantic ranking.
# MAX_RECORDS_PER_SOURCE = 50


# # ============================================================
# # CACHES
# # ============================================================

# _records_cache: list[dict] | None = None

# _graph_cache: nx.Graph | None = None

# _category_records_cache: dict[
#     str,
#     list[dict],
# ] | None = None

# _source_records_cache: dict[
#     str,
#     list[dict],
# ] | None = None


# # ============================================================
# # TEXT / CATEGORY NORMALIZATION
# # ============================================================

# def _clean_text(
#     value: object,
# ) -> str:

#     if value is None:

#         return ""

#     text = str(value)

#     text = re.sub(
#         r"\s+",
#         " ",
#         text,
#     )

#     return text.strip()


# def _normalise(
#     value: object,
# ) -> str:

#     text = _clean_text(
#         value
#     ).lower()

#     text = re.sub(
#         r"[^a-z0-9]+",
#         " ",
#         text,
#     )

#     return text.strip()


# def _category_key(
#     value: object,
# ) -> str:
#     """
#     Convert a category into a stable comparison key.

#     IMPORTANT:
#     This does NOT collapse the LegalBERT categories into
#     broader risk families.

#     Example:

#         "Indemnifications"
#             -> "indemnifications"

#         "Indemnity"
#             -> "indemnity"

#         "Termination"
#             -> "termination"
#     """

#     text = _normalise(
#         value
#     )

#     if not text:

#         return "other"

#     return text


# # ============================================================
# # CATEGORY COMPATIBILITY
# # ============================================================

# def _category_variants(
#     value: object,
# ) -> set[str]:
#     """
#     Generate safe lexical variants for a category.

#     This does NOT replace the original category.

#     It only helps the graph connect labels such as:

#         indemnification
#         indemnifications
#         indemnity

#         termination
#         terminations
#     """

#     key = _category_key(
#         value
#     )

#     if key == "other":

#         return {"other"}

#     variants = {
#         key
#     }

#     # Singular/plural normalization.

#     if key.endswith("ies"):

#         variants.add(
#             key[:-3] + "y"
#         )

#     elif key.endswith("s"):

#         variants.add(
#             key[:-1]
#         )

#     else:

#         variants.add(
#             key + "s"
#         )

#     # Common legal terminology variations.

#     variant_aliases = {

#         "indemnity": {
#             "indemnification",
#             "indemnifications",
#         },

#         "indemnification": {
#             "indemnity",
#             "indemnifications",
#         },

#         "indemnifications": {
#             "indemnity",
#             "indemnification",
#         },

#         "termination": {
#             "terminations",
#         },

#         "terminations": {
#             "termination",
#         },

#         "confidentiality": {
#             "confidential information",
#             "non disclosure",
#             "nondisclosure",
#         },

#         "confidential information": {
#             "confidentiality",
#         },

#         "governing law": {
#             "governing laws",
#             "applicable law",
#             "applicable laws",
#         },

#         "governing laws": {
#             "governing law",
#         },

#         "payment": {
#             "payments",
#             "payment terms",
#         },

#         "payments": {
#             "payment",
#             "payment terms",
#         },

#         "liability": {
#             "limitation of liability",
#             "limitations of liability",
#         },

#         "limitation of liability": {
#             "limitations of liability",
#             "liability",
#         },

#         "limitations of liability": {
#             "limitation of liability",
#             "liability",
#         },
#     }

#     if key in variant_aliases:

#         for variant in variant_aliases[key]:

#             variants.add(
#                 _category_key(
#                     variant
#                 )
#             )

#     return variants


# # ============================================================
# # TEXT HELPERS
# # ============================================================

# def _make_snippet(
#     text: str,
#     max_length: int = 500,
# ) -> str:

#     text = _clean_text(
#         text
#     )

#     if len(text) <= max_length:

#         return text

#     shortened = text[
#         :max_length
#     ].rsplit(
#         " ",
#         1,
#     )[0]

#     return (
#         shortened
#         + "..."
#     )


# def _token_set(
#     text: str,
# ) -> set[str]:

#     return set(

#         token

#         for token in re.findall(
#             r"[a-zA-Z]{2,}",
#             text.lower(),
#         )

#     )


# def _keyword_overlap(
#     query: str,
#     text: str,
# ) -> float:

#     query_tokens = _token_set(
#         query
#     )

#     text_tokens = _token_set(
#         text
#     )

#     if not query_tokens:

#         return 0.0

#     overlap = (
#         query_tokens
#         &
#         text_tokens
#     )

#     return (
#         len(overlap)
#         /
#         len(query_tokens)
#     )


# # ============================================================
# # KNOWLEDGE-BASE LOADING
# # ============================================================

# def _load_records() -> list[dict]:

#     global _records_cache

#     if _records_cache is not None:

#         return _records_cache

#     if not os.path.isfile(
#         KNOWLEDGE_BASE_FILE
#     ):

#         raise RuntimeError(

#             "\nThe unified legal knowledge "
#             "base was not found.\n\n"

#             f"Expected:\n"
#             f"{KNOWLEDGE_BASE_FILE}\n\n"

#             "Run:\n\n"

#             "    python "
#             "build_knowledge_base.py\n"

#         )

#     records: list[dict] = []

#     with open(
#         KNOWLEDGE_BASE_FILE,
#         "r",
#         encoding="utf-8",
#     ) as file:

#         for line_number, line in enumerate(
#             file,
#             start=1,
#         ):

#             line = line.strip()

#             if not line:

#                 continue

#             try:

#                 raw_record = json.loads(
#                     line
#                 )

#             except json.JSONDecodeError:

#                 print(

#                     "[legal_kb] "
#                     "Skipping invalid JSON "
#                     f"on line {line_number}."

#                 )

#                 continue

#             text = _clean_text(
#                 raw_record.get(
#                     "text",
#                     "",
#                 )
#             )

#             if not text:

#                 continue

#             category = _clean_text(
#                 raw_record.get(
#                     "category",
#                     "other",
#                 )
#             )

#             source = _clean_text(
#                 raw_record.get(
#                     "source",
#                     "Unknown",
#                 )
#             )

#             records.append(
#                 {

#                     "source": (

#                         source
#                         if source
#                         else "Unknown"

#                     ),

#                     "source_id": _clean_text(

#                         raw_record.get(

#                             "source_id",

#                             f"record_{line_number}",

#                         )

#                     ),

#                     "title": _clean_text(

#                         raw_record.get(

#                             "title",

#                             "Legal knowledge record",

#                         )

#                     ),

#                     # Preserve original dataset label.
#                     "category": (

#                         category
#                         if category
#                         else "other"

#                     ),

#                     # Internal normalized key only.
#                     "category_key": (

#                         _category_key(
#                             category
#                         )

#                     ),

#                     "text": text,

#                     # Preserve metadata.
#                     "metadata": (

#                         raw_record.get(

#                             "metadata",

#                             {},

#                         )

#                     ),

#                 }
#             )

#     if not records:

#         raise RuntimeError(

#             "The knowledge base contains "
#             "no valid legal records."

#         )

#     _records_cache = records

#     source_counts: dict[
#         str,
#         int,
#     ] = defaultdict(
#         int
#     )

#     for record in records:

#         source_counts[
#             record["source"]
#         ] += 1

#     print(

#         "\n[legal_kb] "
#         "Unified legal knowledge "
#         "base loaded."

#     )

#     print(

#         "[legal_kb] "
#         f"Total records: "
#         f"{len(records):,}"

#     )

#     for source, count in sorted(
#         source_counts.items()
#     ):

#         print(

#             "[legal_kb] "
#             f"{source}: "
#             f"{count:,} records"

#         )

#     return _records_cache


# # ============================================================
# # GRAPH CONSTRUCTION
# # ============================================================

# def _build_graph() -> nx.Graph:

#     records = _load_records()

#     graph = nx.Graph()

#     # --------------------------------------------------------
#     # CATEGORY NODES
#     # --------------------------------------------------------

#     for category_key in {

#         record["category_key"]

#         for record in records

#     }:

#         graph.add_node(

#             f"category::{category_key}",

#             kind="category",

#             label=category_key,

#         )

#     # --------------------------------------------------------
#     # DATASET SOURCE NODES
#     # --------------------------------------------------------

#     for source in {

#         record["source"]

#         for record in records

#     }:

#         graph.add_node(

#             f"source::{source}",

#             kind="source",

#             label=source,

#         )

#     # --------------------------------------------------------
#     # DATASET-SPECIFIC CATEGORY NODES
#     # --------------------------------------------------------

#     for record in records:

#         category_key = (
#             record["category_key"]
#         )

#         category = (
#             record["category"]
#         )

#         source = (
#             record["source"]
#         )

#         category_node = (
#             f"category::{category_key}"
#         )

#         dataset_category_node = (

#             "dataset_category::"
#             f"{source}::"
#             f"{category}"

#         )

#         source_node = (
#             f"source::{source}"
#         )

#         graph.add_node(

#             dataset_category_node,

#             kind="dataset_category",

#             label=category,

#             category_key=category_key,

#             source=source,

#         )

#         graph.add_edge(

#             category_node,

#             dataset_category_node,

#         )

#         graph.add_edge(

#             dataset_category_node,

#             source_node,

#         )

#     # --------------------------------------------------------
#     # LEXICAL CATEGORY RELATIONSHIPS
#     # --------------------------------------------------------

#     category_nodes = [

#         node

#         for node, data in graph.nodes(
#             data=True
#         )

#         if data.get(
#             "kind"
#         ) == "category"

#     ]

#     category_keys = {

#         node.split(
#             "category::",
#             1
#         )[1]: node

#         for node in category_nodes

#     }

#     for category_key, category_node in (
#         category_keys.items()
#     ):

#         variants = _category_variants(
#             category_key
#         )

#         for variant in variants:

#             if variant == category_key:

#                 continue

#             related_node = category_keys.get(
#                 variant
#             )

#             if related_node is not None:

#                 graph.add_edge(

#                     category_node,

#                     related_node,

#                     relation="category_variant",

#                 )

#     return graph


# def _graph() -> nx.Graph:

#     global _graph_cache

#     if _graph_cache is None:

#         _graph_cache = (
#             _build_graph()
#         )

#         print(

#             "[legal_kb] "
#             "Lightweight GraphRAG "
#             "graph created."

#         )

#         print(

#             "[legal_kb] "
#             f"Graph nodes: "
#             f"{_graph_cache.number_of_nodes():,}"

#         )

#         print(

#             "[legal_kb] "
#             f"Graph edges: "
#             f"{_graph_cache.number_of_edges():,}"

#         )

#     return _graph_cache


# # ============================================================
# # RECORD GROUPING
# # ============================================================

# def _build_record_groups() -> None:

#     global _category_records_cache
#     global _source_records_cache

#     if (

#         _category_records_cache is not None

#         and

#         _source_records_cache is not None

#     ):

#         return

#     records = _load_records()

#     category_groups = defaultdict(
#         list
#     )

#     source_groups = defaultdict(
#         list
#     )

#     for record in records:

#         category_groups[
#             record["category_key"]
#         ].append(
#             record
#         )

#         source_groups[
#             record["source"]
#         ].append(
#             record
#         )

#     _category_records_cache = dict(
#         category_groups
#     )

#     _source_records_cache = dict(
#         source_groups
#     )


# # ============================================================
# # GRAPH-BASED CATEGORY TRAVERSAL
# # ============================================================

# def _graph_categories(
#     predicted_type: str | None,
# ) -> list[str]:

#     graph = _graph()

#     if not predicted_type:

#         return []

#     predicted_key = _category_key(
#         predicted_type
#     )

#     category_node = (
#         f"category::{predicted_key}"
#     )

#     categories: list[str] = []

#     # --------------------------------------------------------
#     # Exact predicted category.
#     # --------------------------------------------------------

#     if category_node in graph:

#         categories.append(
#             predicted_key
#         )

#         # ----------------------------------------------------
#         # Direct graph neighbours.
#         # ----------------------------------------------------

#         for neighbour in graph.neighbors(
#             category_node
#         ):

#             data = graph.nodes[
#                 neighbour
#             ]

#             kind = data.get(
#                 "kind"
#             )

#             if kind == "dataset_category":

#                 related_key = _category_key(

#                     data.get(
#                         "label",
#                         "",
#                     )

#                 )

#                 if (
#                     related_key
#                     not in categories
#                 ):

#                     categories.append(
#                         related_key
#                     )

#             elif kind == "category":

#                 related_key = _category_key(

#                     data.get(
#                         "label",
#                         "",
#                     )

#                 )

#                 if (
#                     related_key
#                     not in categories
#                 ):

#                     categories.append(
#                         related_key
#                     )

#     else:

#         # ----------------------------------------------------
#         # Safe lexical fallback.
#         # ----------------------------------------------------

#         variants = _category_variants(
#             predicted_type
#         )

#         for variant in variants:

#             variant_node = (
#                 f"category::{variant}"
#             )

#             if variant_node not in graph:

#                 continue

#             if variant not in categories:

#                 categories.append(
#                     variant
#                 )

#     return categories


# # ============================================================
# # GRAPH-BASED CANDIDATE SELECTION
# # ============================================================

# def _select_graph_candidates(
#     query: str,
#     predicted_type: str | None,
# ) -> list[dict]:
#     """
#     Select a small candidate pool before semantic scoring.

#     IMPORTANT:

#     1. LegalBERT's predicted category is used to narrow
#        the search.

#     2. The ORIGINAL clause text is used for keyword
#        relevance while constructing the candidate pool.

#     3. Candidates are collected across relevant categories
#        and sources.

#     4. No source is forced into the final results.

#     5. Final ranking is performed by semantic similarity
#        against the ORIGINAL clause text.

#     This means:

#         classification
#              +
#         original clause
#              ↓
#         candidate generation
#              ↓
#         semantic ranking
#              ↓
#         top-k relevant records
#     """

#     _build_record_groups()

#     categories = _graph_categories(
#         predicted_type
#     )

#     selected: list[dict] = []

#     seen_ids: set[str] = set()

#     # Track how many records from each source have entered
#     # the candidate pool.
#     source_counts: dict[
#         str,
#         int,
#     ] = defaultdict(
#         int
#     )

#     # --------------------------------------------------------
#     # PRIORITY 1:
#     # Search records belonging to categories connected to
#     # the LegalBERT prediction.
#     #
#     # IMPORTANT:
#     # Do NOT simply take 40 records from LEDGAR and stop.
#     # We rank records by the ORIGINAL clause text and keep
#     # source diversity during candidate construction.
#     # --------------------------------------------------------

#     category_ranked_records: list[dict] = []

#     for category in categories:

#         records = (
#             _category_records_cache.get(
#                 category,
#                 [],
#             )
#         )

#         ranked = sorted(

#             records,

#             key=lambda record: (

#                 _keyword_overlap(

#                     query,

#                     record["title"]
#                     + " "
#                     + record["category"]
#                     + " "
#                     + record["text"],

#                 ),

#                 len(
#                     record["text"]
#                 ),

#             ),

#             reverse=True,

#         )

#         # Keep the best category-specific records available.
#         category_ranked_records.extend(

#             ranked[
#                 :MAX_RECORDS_PER_CATEGORY
#             ]

#         )

#     # Re-rank all category candidates together using the
#     # ORIGINAL clause text.
#     category_ranked_records.sort(

#         key=lambda record: (

#             _keyword_overlap(

#                 query,

#                 record["title"]
#                 + " "
#                 + record["category"]
#                 + " "
#                 + record["text"],

#             ),

#         ),

#         reverse=True,

#     )

#     for record in category_ranked_records:

#         source_id = (
#             record["source_id"]
#         )

#         if source_id in seen_ids:

#             continue

#         source = (
#             record["source"]
#         )

#         # Prevent one dataset from occupying the complete
#         # candidate pool.
#         if (

#             source_counts[source]
#             >= MAX_RECORDS_PER_SOURCE

#         ):

#             continue

#         selected.append(
#             record
#         )

#         seen_ids.add(
#             source_id
#         )

#         source_counts[source] += 1

#         if (
#             len(selected)
#             >= MAX_GRAPH_CANDIDATES
#         ):

#             break

#     # --------------------------------------------------------
#     # PRIORITY 2:
#     # If category-based candidates are insufficient, search
#     # the complete KB using the ORIGINAL clause text.
#     #
#     # This is particularly important for CUAD and MAUD because
#     # their category vocabulary may not exactly match LEDGAR.
#     # --------------------------------------------------------

#     if len(selected) < MAX_GRAPH_CANDIDATES:

#         all_records = _load_records()

#         ranked_all = sorted(

#             all_records,

#             key=lambda record: (

#                 _keyword_overlap(

#                     query,

#                     record["title"]
#                     + " "
#                     + record["category"]
#                     + " "
#                     + record["text"],

#                 ),

#             ),

#             reverse=True,

#         )

#         for record in ranked_all:

#             source_id = (
#                 record["source_id"]
#             )

#             if source_id in seen_ids:

#                 continue

#             source = (
#                 record["source"]
#             )

#             if (

#                 source_counts[source]
#                 >= MAX_RECORDS_PER_SOURCE

#             ):

#                 continue

#             selected.append(
#                 record
#             )

#             seen_ids.add(
#                 source_id
#             )

#             source_counts[source] += 1

#             if (
#                 len(selected)
#                 >= MAX_GRAPH_CANDIDATES
#             ):

#                 break

#     return selected[
#         :MAX_GRAPH_CANDIDATES
#     ]


# # ============================================================
# # SEMANTIC SCORING
# # ============================================================

# def _semantic_score(
#     query: str,
#     candidates: list[dict],
# ) -> list[tuple[float, dict]]:

#     if not candidates:

#         return []

#     encoder = get_encoder()

#     # --------------------------------------------------------
#     # IMPORTANT:
#     # The ORIGINAL CLAUSE is encoded here.
#     #
#     # The predicted category is NOT the semantic query.
#     # --------------------------------------------------------

#     query_vector = np.asarray(

#         encoder.encode(
#             query
#         ),

#         dtype=np.float32,

#     )

#     candidate_texts = [

#         (

#             record["title"]
#             + " "
#             + record["category"]
#             + " "
#             + record["text"]

#         )

#         for record in candidates

#     ]

#     if hasattr(
#         encoder,
#         "encode_batch",
#     ):

#         candidate_vectors = (

#             encoder.encode_batch(

#                 candidate_texts,

#                 batch_size=16,

#                 show_progress=False,

#             )

#         )

#     else:

#         candidate_vectors = [

#             encoder.encode(
#                 text
#             )

#             for text in candidate_texts

#         ]

#     scored: list[
#         tuple[
#             float,
#             dict,
#         ]
#     ] = []

#     for record, vector in zip(

#         candidates,

#         candidate_vectors,

#     ):

#         semantic_similarity = (

#             cosine_similarity(

#                 query_vector,

#                 np.asarray(

#                     vector,

#                     dtype=np.float32,

#                 ),

#             )

#         )

#         keyword_score = (

#             _keyword_overlap(

#                 query,

#                 record["title"]
#                 + " "
#                 + record["category"]
#                 + " "
#                 + record["text"],

#             )

#         )

#         # ----------------------------------------------------
#         # Final relevance:
#         #
#         # 80% semantic similarity
#         # 20% keyword overlap
#         #
#         # No source-specific bonus is applied.
#         # ----------------------------------------------------

#         relevance = (

#             0.80
#             *
#             semantic_similarity

#             +

#             0.20
#             *
#             keyword_score

#         )

#         relevance = max(

#             0.0,

#             min(

#                 1.0,

#                 relevance,

#             ),

#         )

#         scored.append(

#             (
#                 relevance,
#                 record,
#             )

#         )

#     scored.sort(

#         key=lambda item: item[0],

#         reverse=True,

#     )

#     return scored


# # ============================================================
# # RETRIEVAL
# # ============================================================

# def retrieve(
#     query: str,
#     clause_type: str | None = None,
#     top_k: int = 5,
# ) -> RetrievalResult:
#     """
#     Retrieve the most relevant legal records.

#     Retrieval uses BOTH:

#         1. LegalBERT predicted category
#         2. Original clause text

#     The category performs graph-based narrowing.

#     The original clause performs keyword filtering and
#     semantic ranking.

#     CUAD / MAUD are NOT forced into the final results.
#     """

#     query = _clean_text(
#         query
#     )

#     if not query:

#         raise ValueError(
#             "The retrieval query is empty."
#         )

#     if top_k < 1:

#         raise ValueError(
#             "top_k must be at least 1."
#         )

#     # --------------------------------------------------------
#     # Build graph once.
#     # --------------------------------------------------------

#     _graph()

#     # --------------------------------------------------------
#     # GraphRAG candidate selection.
#     # --------------------------------------------------------

#     candidates = (

#         _select_graph_candidates(

#             query,

#             clause_type,

#         )

#     )

#     # Print source distribution so we can verify that
#     # candidate generation is not dominated by one dataset.

#     candidate_source_counts = defaultdict(
#         int
#     )

#     for record in candidates:

#         candidate_source_counts[
#             record["source"]
#         ] += 1

#     print(

#         "[legal_kb] "
#         f"Graph selected "
#         f"{len(candidates)} candidate "
#         f"records for "
#         f"'{clause_type}'."

#     )

#     print(

#         "[legal_kb] "
#         "Candidate source distribution: "

#         + ", ".join(

#             f"{source}={count}"

#             for source, count in sorted(
#                 candidate_source_counts.items()
#             )

#         )

#     )

#     # --------------------------------------------------------
#     # Semantic scoring.
#     # --------------------------------------------------------

#     scored = _semantic_score(

#         query,

#         candidates,

#     )

#     hits: list[
#         RetrievalHit
#     ] = []

#     predicted_key = _category_key(
#         clause_type
#     )

#     for relevance, record in scored[
#         :top_k
#     ]:

#         source = (
#             record["source"]
#         )

#         category = (
#             record["category"]
#         )

#         category_key = (
#             record["category_key"]
#         )

#         graph_path = [

#             predicted_key,

#             category_key,

#             source,

#         ]

#         hits.append(

#             RetrievalHit(

#                 source_id=(

#                     record[
#                         "source_id"
#                     ]

#                 ),

#                 title=(

#                     f"{record['title']} "
#                     f"[{source}]"

#                 ),

#                 snippet=(

#                     _make_snippet(

#                         record[
#                             "text"
#                         ]

#                     )

#                 ),

#                 relevance=round(

#                     relevance,

#                     4,

#                 ),

#                 graph_path=graph_path,

#             )

#         )

#     return RetrievalResult(

#         query=query,

#         results=hits,

#     )


# # ============================================================
# # STANDALONE TEST
# # ============================================================

# if __name__ == "__main__":

#     print(

#         "\nTesting lightweight "
#         "GraphRAG retrieval...\n"

#     )

#     result = retrieve(

#         query=(

#             "The Vendor shall "
#             "indemnify, defend, "
#             "and hold harmless "
#             "the Client from "
#             "all claims and "
#             "losses."

#         ),

#         clause_type=(

#             "Indemnifications"

#         ),

#         top_k=5,

#     )

#     print(

#         result.model_dump_json(

#             indent=2

#         )

#     )

#     print(

#         "\nOK — lightweight "
#         "GraphRAG retrieval "
#         "completed."

#     )

"""
Stage 3 (Veda): Legal knowledge retrieval.

Architecture:

    Classification
          |
          v
    Predicted LEDGAR category
          |
          +----------------------+
          |                      |
          v                      v
    Category awareness      Original clause
          |                      |
          +----------+-----------+
                     |
                     v
              FAISS retrieval
                     |
                     v
             Lightweight rerank
                     |
                     v
                Evidence

Knowledge base:
    LEDGAR + CUAD + MAUD

FAISS contains precomputed LegalBERT embeddings for all
knowledge-base records.

The embeddings are NOT regenerated during every contract analysis.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import faiss
import networkx as nx
import numpy as np

from embeddings import (
    get_encoder,
    cosine_similarity,
)

from interfaces import (
    RetrievalHit,
    RetrievalResult,
)


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

KNOWLEDGE_BASE_FILE = os.path.join(
    KB_DIR,
    "legal_records.jsonl",
)

FAISS_INDEX_FILE = os.path.join(
    KB_DIR,
    "legal_faiss.index",
)

FAISS_METADATA_FILE = os.path.join(
    KB_DIR,
    "legal_faiss_metadata.json",
)


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

# Number of FAISS candidates retrieved globally.
#
# This is cheap because FAISS searches precomputed vectors.
FAISS_TOP_K = 200


# Number of final records returned to the pipeline.
DEFAULT_TOP_K = 5


# Maximum records retained after reranking.
MAX_RERANK_CANDIDATES = 80


# Relevance weights.
#
# Semantic similarity remains dominant.
SEMANTIC_WEIGHT = 0.80
KEYWORD_WEIGHT = 0.20


# ============================================================
# CACHES
# ============================================================

_records_cache: list[dict] | None = None

_faiss_index = None

_graph_cache: nx.Graph | None = None

_category_records_cache: dict[
    str,
    list[dict],
] | None = None


# ============================================================
# TEXT HELPERS
# ============================================================

def _clean_text(
    value: object,
) -> str:

    if value is None:
        return ""

    text = str(
        value
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _normalise(
    value: object,
) -> str:

    text = _clean_text(
        value
    ).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return text.strip()


def _category_key(
    value: object,
) -> str:

    text = _normalise(
        value
    )

    if not text:
        return "other"

    return text


def _make_snippet(
    text: str,
    max_length: int = 500,
) -> str:

    text = _clean_text(
        text
    )

    if len(text) <= max_length:
        return text

    shortened = text[
        :max_length
    ].rsplit(
        " ",
        1,
    )[0]

    return shortened + "..."


def _token_set(
    text: str,
) -> set[str]:

    return set(

        token

        for token in re.findall(
            r"[a-zA-Z]{2,}",
            text.lower(),
        )

    )


def _keyword_overlap(
    query: str,
    text: str,
) -> float:

    query_tokens = _token_set(
        query
    )

    text_tokens = _token_set(
        text
    )

    if not query_tokens:
        return 0.0

    overlap = (
        query_tokens
        &
        text_tokens
    )

    return (
        len(overlap)
        /
        len(query_tokens)
    )


# ============================================================
# CATEGORY VARIANTS
# ============================================================

def _category_variants(
    value: object,
) -> set[str]:

    key = _category_key(
        value
    )

    if key == "other":
        return {"other"}

    variants = {
        key
    }


    # Singular / plural variants
    if key.endswith("ies"):

        variants.add(
            key[:-3] + "y"
        )

    elif key.endswith("s"):

        variants.add(
            key[:-1]
        )

    else:

        variants.add(
            key + "s"
        )


    # Known legal terminology variants
    variant_aliases = {

        "indemnity": {
            "indemnification",
            "indemnifications",
        },

        "indemnification": {
            "indemnity",
            "indemnifications",
        },

        "indemnifications": {
            "indemnity",
            "indemnification",
        },

        "termination": {
            "terminations",
        },

        "terminations": {
            "termination",
        },

        "confidentiality": {
            "confidential information",
            "non disclosure",
            "nondisclosure",
        },

        "confidential information": {
            "confidentiality",
        },

        "governing law": {
            "governing laws",
            "applicable law",
            "applicable laws",
        },

        "governing laws": {
            "governing law",
        },

        "payment": {
            "payments",
            "payment terms",
        },

        "payments": {
            "payment",
            "payment terms",
        },

        "liability": {
            "limitation of liability",
            "limitations of liability",
        },

        "limitation of liability": {
            "limitations of liability",
            "liability",
        },

        "limitations of liability": {
            "limitation of liability",
            "liability",
        },

    }


    if key in variant_aliases:

        for variant in variant_aliases[
            key
        ]:

            variants.add(
                _category_key(
                    variant
                )
            )


    return variants


# ============================================================
# LOAD RECORDS
# ============================================================

def _load_records() -> list[dict]:

    global _records_cache

    if _records_cache is not None:

        return _records_cache


    if not os.path.isfile(
        KNOWLEDGE_BASE_FILE
    ):

        raise RuntimeError(

            "\nThe unified legal knowledge "
            "base was not found.\n\n"

            f"Expected:\n"
            f"{KNOWLEDGE_BASE_FILE}\n\n"

            "Run:\n\n"

            "    python build_knowledge_base.py\n"

        )


    records = []


    with open(
        KNOWLEDGE_BASE_FILE,
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

                raw_record = json.loads(
                    line
                )

            except json.JSONDecodeError:

                print(

                    "[legal_kb] "
                    "Skipping invalid JSON "
                    f"on line {line_number}."

                )

                continue


            text = _clean_text(
                raw_record.get(
                    "text",
                    "",
                )
            )

            if not text:
                continue


            category = _clean_text(
                raw_record.get(
                    "category",
                    "other",
                )
            )


            source = _clean_text(
                raw_record.get(
                    "source",
                    "Unknown",
                )
            )


            records.append(

                {

                    "source": (
                        source
                        if source
                        else "Unknown"
                    ),

                    "source_id": _clean_text(
                        raw_record.get(
                            "source_id",
                            f"record_{line_number}",
                        )
                    ),

                    "title": _clean_text(
                        raw_record.get(
                            "title",
                            "Legal knowledge record",
                        )
                    ),

                    "category": (
                        category
                        if category
                        else "other"
                    ),

                    "category_key": (
                        _category_key(
                            category
                        )
                    ),

                    "text": text,

                    "metadata": (
                        raw_record.get(
                            "metadata",
                            {},
                        )
                    ),

                }

            )


    if not records:

        raise RuntimeError(
            "The knowledge base contains "
            "no valid legal records."
        )


    _records_cache = records


    source_counts = defaultdict(
        int
    )

    for record in records:

        source_counts[
            record["source"]
        ] += 1


    print(
        "\n[legal_kb] "
        "Unified legal knowledge base loaded."
    )

    print(
        "[legal_kb] "
        f"Total records: {len(records):,}"
    )


    for source, count in sorted(
        source_counts.items()
    ):

        print(

            "[legal_kb] "
            f"{source}: {count:,} records"

        )


    return _records_cache


# ============================================================
# LOAD FAISS
# ============================================================

def _load_faiss():

    global _faiss_index


    if _faiss_index is not None:

        return _faiss_index


    if not os.path.isfile(
        FAISS_INDEX_FILE
    ):

        raise RuntimeError(

            "\nFAISS index was not found.\n\n"

            f"Expected:\n"
            f"{FAISS_INDEX_FILE}\n\n"

            "Build it once using:\n\n"

            "    python build_faiss_index.py\n"

        )


    if not os.path.isfile(
        FAISS_METADATA_FILE
    ):

        raise RuntimeError(

            "\nFAISS metadata was not found.\n\n"

            f"Expected:\n"
            f"{FAISS_METADATA_FILE}\n\n"

            "Run:\n\n"

            "    python build_faiss_index.py\n"

        )


    print(
        "\n[legal_kb] Loading FAISS index..."
    )


    _faiss_index = faiss.read_index(
        FAISS_INDEX_FILE
    )


    records = _load_records()


    if _faiss_index.ntotal != len(
        records
    ):

        raise RuntimeError(

            "\nFAISS index / knowledge-base mismatch.\n\n"

            f"FAISS records: "
            f"{_faiss_index.ntotal:,}\n"

            f"Knowledge-base records: "
            f"{len(records):,}\n\n"

            "Rebuild the index using:\n\n"

            "    python build_faiss_index.py\n"

        )


    print(

        "[legal_kb] "
        f"FAISS index loaded: "
        f"{_faiss_index.ntotal:,} vectors."

    )


    return _faiss_index


# ============================================================
# GRAPH
# ============================================================

def _build_graph() -> nx.Graph:

    records = _load_records()

    graph = nx.Graph()


    # Category nodes
    for category_key in {

        record[
            "category_key"
        ]

        for record
        in records

    }:

        graph.add_node(

            f"category::{category_key}",

            kind="category",

            label=category_key,

        )


    # Dataset source nodes
    for source in {

        record[
            "source"
        ]

        for record
        in records

    }:

        graph.add_node(

            f"source::{source}",

            kind="source",

            label=source,

        )


    # Dataset-category nodes
    for record in records:

        category_key = (
            record[
                "category_key"
            ]
        )

        category = (
            record[
                "category"
            ]
        )

        source = (
            record[
                "source"
            ]
        )


        category_node = (
            f"category::{category_key}"
        )


        dataset_category_node = (

            "dataset_category::"
            f"{source}::"
            f"{category}"

        )


        source_node = (
            f"source::{source}"
        )


        graph.add_node(

            dataset_category_node,

            kind="dataset_category",

            label=category,

            category_key=category_key,

            source=source,

        )


        graph.add_edge(

            category_node,

            dataset_category_node,

        )


        graph.add_edge(

            dataset_category_node,

            source_node,

        )


    # Lexical category relationships
    category_nodes = [

        node

        for node, data
        in graph.nodes(
            data=True
        )

        if data.get(
            "kind"
        ) == "category"

    ]


    category_keys = {

        node.split(
            "category::",
            1
        )[1]: node

        for node
        in category_nodes

    }


    for category_key, category_node in (
        category_keys.items()
    ):

        variants = _category_variants(
            category_key
        )


        for variant in variants:

            if variant == category_key:
                continue


            related_node = (
                category_keys.get(
                    variant
                )
            )


            if related_node is not None:

                graph.add_edge(

                    category_node,

                    related_node,

                    relation="category_variant",

                )


    return graph


def _graph() -> nx.Graph:

    global _graph_cache


    if _graph_cache is None:

        _graph_cache = (
            _build_graph()
        )


        print(
            "[legal_kb] "
            "Lightweight GraphRAG graph created."
        )


        print(

            "[legal_kb] "
            f"Graph nodes: "
            f"{_graph_cache.number_of_nodes():,}"

        )


        print(

            "[legal_kb] "
            f"Graph edges: "
            f"{_graph_cache.number_of_edges():,}"

        )


    return _graph_cache


# ============================================================
# BUILD CATEGORY GROUPS
# ============================================================

def _build_record_groups():

    global _category_records_cache


    if (
        _category_records_cache
        is not None
    ):

        return


    records = _load_records()


    category_groups = defaultdict(
        list
    )


    for record in records:

        category_groups[
            record["category_key"]
        ].append(
            record
        )


    _category_records_cache = dict(
        category_groups
    )


# ============================================================
# GRAPH CATEGORY CONNECTION
# ============================================================

def _graph_categories(
    predicted_type: str | None,
) -> set[str]:

    graph = _graph()


    if not predicted_type:

        return set()


    predicted_key = (
        _category_key(
            predicted_type
        )
    )


    category_node = (
        f"category::{predicted_key}"
    )


    categories = set()


    if category_node in graph:

        categories.add(
            predicted_key
        )


        for neighbour in graph.neighbors(
            category_node
        ):

            data = graph.nodes[
                neighbour
            ]


            kind = data.get(
                "kind"
            )


            if kind == "dataset_category":

                categories.add(

                    _category_key(
                        data.get(
                            "label",
                            "",
                        )
                    )

                )


            elif kind == "category":

                categories.add(

                    _category_key(
                        data.get(
                            "label",
                            "",
                        )
                    )

                )


    else:

        for variant in _category_variants(
            predicted_type
        ):

            variant_node = (
                f"category::{variant}"
            )


            if variant_node in graph:

                categories.add(
                    variant
                )


    return categories


# ============================================================
# FAISS RETRIEVAL
# ============================================================

def _faiss_candidates(
    query: str,
    predicted_type: str | None,
) -> list[tuple[float, dict]]:

    index = _load_faiss()

    records = _load_records()

    encoder = get_encoder()


    # --------------------------------------------------------
    # Encode ONLY the current clause.
    # --------------------------------------------------------

    query_vector = np.asarray(

        encoder.encode(
            query
        ),

        dtype=np.float32,

    ).reshape(
        1,
        -1,
    )


    # --------------------------------------------------------
    # Normalize query.
    # --------------------------------------------------------

    faiss.normalize_L2(
        query_vector
    )


    # --------------------------------------------------------
    # One FAISS search.
    #
    # This searches the complete 33k-vector index
    # without regenerating any candidate embeddings.
    # --------------------------------------------------------

    search_k = min(

        FAISS_TOP_K,

        index.ntotal,

    )


    similarities, indices = (
        index.search(
            query_vector,
            search_k,
        )
    )


    graph_categories = (
        _graph_categories(
            predicted_type
        )
    )


    candidates = []


    for similarity, index_id in zip(

        similarities[0],

        indices[0],

    ):

        if index_id < 0:
            continue


        record = records[
            int(index_id)
        ]


        semantic_similarity = float(
            similarity
        )


        full_text = (

            record["title"]
            + " "
            + record["category"]
            + " "
            + record["text"]

        )


        keyword_score = (
            _keyword_overlap(
                query,
                full_text,
            )
        )


        category_key = (
            record["category_key"]
        )


        # ----------------------------------------------------
        # Category awareness.
        #
        # This is NOT a risk score.
        # It is only retrieval prioritization.
        # ----------------------------------------------------

        category_match = (

            1.0

            if (
                category_key
                in graph_categories
            )

            else 0.0

        )


        # ----------------------------------------------------
        # Final retrieval relevance.
        #
        # Semantic similarity remains dominant.
        #
        # Category match gets a small bonus because the
        # classifier already identified the clause type.
        # ----------------------------------------------------

        relevance = (

            0.75
            * semantic_similarity

            +

            0.15
            * keyword_score

            +

            0.10
            * category_match

        )


        relevance = max(

            0.0,

            min(
                1.0,
                relevance,
            ),

        )


        candidates.append(

            (
                relevance,
                record,
                semantic_similarity,
                keyword_score,
                category_match,
            )

        )


    candidates.sort(

        key=lambda item: item[0],

        reverse=True,

    )


    return candidates[
        :MAX_RERANK_CANDIDATES
    ]


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    query: str,
    clause_type: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> RetrievalResult:

    """
    Retrieve evidence from:

        LEDGAR
        CUAD
        MAUD

    using:

        LegalBERT embedding
              +
        FAISS vector search
              +
        keyword relevance
              +
        predicted-category awareness
    """

    query = _clean_text(
        query
    )


    if not query:

        raise ValueError(
            "The retrieval query is empty."
        )


    if top_k < 1:

        raise ValueError(
            "top_k must be at least 1."
        )


    # Load once.
    _load_records()
    _graph()
    _load_faiss()


    # --------------------------------------------------------
    # Retrieve candidates.
    # --------------------------------------------------------

    candidates = _faiss_candidates(

        query,

        clause_type,

    )


    print(

        "[legal_kb] "
        f"FAISS retrieved "
        f"{len(candidates)} candidate records "
        f"for '{clause_type}'."

    )


    # --------------------------------------------------------
    # Source distribution.
    # --------------------------------------------------------

    source_counts = defaultdict(
        int
    )


    for _, record, *_ in candidates:

        source_counts[
            record["source"]
        ] += 1


    print(

        "[legal_kb] "
        "FAISS candidate source distribution: "

        +

        ", ".join(

            f"{source}={count}"

            for source, count
            in sorted(
                source_counts.items()
            )

        )

    )


    # --------------------------------------------------------
    # Final results.
    # --------------------------------------------------------

    hits = []


    predicted_key = (
        _category_key(
            clause_type
        )
    )


    for (

        relevance,

        record,

        semantic_similarity,

        keyword_score,

        category_match,

    ) in candidates[:top_k]:


        source = (
            record["source"]
        )


        category = (
            record["category"]
        )


        category_key = (
            record["category_key"]
        )


        graph_path = [

            predicted_key,

            category_key,

            source,

        ]


        hits.append(

            RetrievalHit(

                source_id=(
                    record[
                        "source_id"
                    ]
                ),

                title=(

                    f"{record['title']} "
                    f"[{source}]"

                ),

                snippet=(

                    _make_snippet(
                        record["text"]
                    )

                ),

                relevance=round(
                    relevance,
                    4,
                ),

                graph_path=graph_path,

            )

        )


    return RetrievalResult(

        query=query,

        results=hits,

    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nTesting FAISS legal retrieval...\n"
    )


    result = retrieve(

        query=(

            "The Vendor shall "
            "indemnify, defend, "
            "and hold harmless "
            "the Client from "
            "all claims and losses."

        ),

        clause_type=(
            "Indemnifications"
        ),

        top_k=5,

    )


    print(

        result.model_dump_json(
            indent=2
        )

    )


    print(
        "\nOK — FAISS retrieval completed."
    )