"""
Compatibility mapping for the fine-tuned LegalBERT classifier.

The current LegalBERT model was fine-tuned directly on the final set of
LEDGAR categories selected through semantic mapping against CUAD + ACORD.

Therefore:

    LegalBERT prediction
            |
            v
    final LEDGAR category

There is NO conversion back to the old 7-category taxonomy.

The mapping is intentionally identity-based because the model's output labels
are already the final categories used by the project.

The authoritative label list is also stored inside:

    legalbert_finetuned/selected_categories.json

and in:

    legalbert_finetuned/config.json
"""


# ============================================================
# FINAL LABELS USED BY THE CURRENT LEGALBERT MODEL
# ============================================================

FINAL_LEDGAR_CATEGORIES = [
    "Agreements",
    "Amendments",
    "Applicable Laws",
    "Assignments",
    "Assigns",
    "Authority",
    "Authorizations",
    "Change In Control",
    "Cooperation",
    "Costs",
    "Defined Terms",
    "Effective Dates",
    "Expenses",
    "Governing Laws",
    "Indemnifications",
    "Indemnity",
    "Insurances",
    "Intellectual Property",
    "Jurisdictions",
    "Liens",
    "Litigations",
    "No Waivers",
    "Non-Disparagement",
    "Participations",
    "Sales",
    "Sanctions",
    "Severability",
    "Terminations",
    "Terms",
    "Titles",
    "Transactions With Affiliates",
    "Use Of Proceeds",
    "Waiver Of Jury Trials",
    "Warranties",
]


# ============================================================
# IDENTITY MAPPING
# ============================================================
#
# The model already predicts these final LEDGAR labels.
#
# Example:
#
#     "Indemnifications" -> "Indemnifications"
#     "Warranties"       -> "Warranties"
#     "Sanctions"        -> "Sanctions"
#
# No information is collapsed into "other".
# ============================================================

LEDGAR_TO_OURS: dict[str, str] = {
    category: category
    for category in FINAL_LEDGAR_CATEGORIES
}


def map_label(
    ledgar_label: str,
) -> str:
    """
    Return the final LegalBERT LEDGAR category unchanged.

    Raises an error if an unexpected label is supplied instead of silently
    converting it to the obsolete 'other' category.
    """

    if ledgar_label not in LEDGAR_TO_OURS:

        raise ValueError(
            "\nUnexpected LegalBERT label: "
            f"{ledgar_label}\n\n"

            "This label is not part of the 34 final "
            "LEDGAR categories used by the current "
            "fine-tuned model.\n\n"

            "Check legalbert_finetuned/"
            "selected_categories.json."
        )

    return LEDGAR_TO_OURS[
        ledgar_label
    ]


# """Maps LEDGAR's fine-grained clause categories down to our 7-category
# taxonomy (see CLAUSE_TYPES in classifier.py / docs/INTERFACES.md section 2).

# LEDGAR ships ~100 categories scraped from real SEC/EDGAR contracts. The names
# below are written from memory of the LexGLUE/LEDGAR label set, not verified
# against a live download (no internet in the environment that wrote this) —
# run `python inspect_ledgar_labels.py` FIRST and diff its output against the
# keys here before training. Dataset label strings occasionally differ in
# capitalization/pluralization between HF dataset versions.

# Any LEDGAR label not listed here maps to "other" by default — intentional:
# it gives the model realistic negative ("not one of our 6 tracked types")
# examples instead of silently dropping 90+ categories of training data.
# """

# LEDGAR_TO_OURS: dict[str, str] = {
#     # indemnification
#     "Indemnifications": "indemnification",
#     "Indemnity": "indemnification",

#     # termination
#     "Terminations": "termination",
#     "Termination": "termination",
#     "Survival": "termination",

#     # confidentiality
#     "Confidentiality": "confidentiality",
#     "Non-Disclosure": "confidentiality",
#     "Nondisclosure": "confidentiality",

#     # governing_law
#     "Governing Laws": "governing_law",
#     "Governing Law": "governing_law",
#     "Jurisdictions": "governing_law",
#     "Submission To Jurisdiction": "governing_law",
#     "Consent To Jurisdiction": "governing_law",
#     "Venues": "governing_law",
#     "Arbitration": "governing_law",

#     # payment_terms
#     "Payments": "payment_terms",
#     "Fees": "payment_terms",
#     "Expenses": "payment_terms",
#     "Costs": "payment_terms",

#     # limitation_of_liability
#     "Limitation Of Liability": "limitation_of_liability",
#     "Liability": "limitation_of_liability",
#     "Remedies": "limitation_of_liability",
#     "Damages": "limitation_of_liability",
# }


# def map_label(ledgar_label: str) -> str:
#     """Fine LEDGAR category -> one of our 7 coarse types. Unknown -> 'other'."""
#     return LEDGAR_TO_OURS.get(ledgar_label, "other")
