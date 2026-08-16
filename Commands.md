## Required Generated Files

Some large files are excluded from GitHub through `.gitignore`. After cloning the repository, the project should have the following structure:

```text
legalcontractreview_Veda/
│
├── data/
│   ├── knowledge_base/
│   │   ├── legal_faiss_metadata.json
│   │   ├── legal_faiss.index
│   │   └── legal_records.jsonl
│   │
│   ├── uploads/
│   ├── sample_document.json
│   └── test_contract.docx
│
├── legalbert_finetuned/
│
├── training_output/
│
├── docs/
├── classifier.py
├── embeddings.py
├── legal_kb.py
├── build_faiss_index.py
├── pipeline.py
├── app.py
└── ...

1. Obtain legalbert_finetuned/

The fine-tuned LegalBERT model is not stored in GitHub because of its size.

Run the LegalBERT model notebook provided with the project.

The notebook:

Downloads/loads the required LegalBERT model and LEDGAR dataset.
Fine-tunes LegalBERT on the original 100 LEDGAR clause categories.
Saves the trained model and tokenizer.
Creates the required:
legalbert_finetuned/

After running the notebook, make sure the folder is placed in the project root:

project_folder/
└── legalbert_finetuned/

The classifier loads the model directly from this folder.

Important: The classifier expects the fine-tuned model to contain the same 100-category LEDGAR label mapping used during training.

2. Obtain the Knowledge Base and FAISS Index

The retrieval system uses the generated knowledge base and FAISS index:
###important###
data/knowledge_base/
├── legal_records.jsonl
├── legal_faiss.index
└── legal_faiss_metadata.json

Build the knowledge base first, then build the FAISS index.

python build_knowledge_base.py
python build_faiss_index.py

The resulting files must be placed under:

data/knowledge_base/

3. Final Setup

Once the generated files are available, verify:

legalcontractreview_Veda/
│
├── data/
│   └── knowledge_base/
│       ├── legal_faiss_metadata.json
│       ├── legal_faiss.index
│       └── legal_records.jsonl
│
├── legalbert_finetuned/
│
└── application code...

Then run:

streamlit run app.py

You do not need to retrain LegalBERT every time the application is run. The notebook is only required when the legalbert_finetuned/ folder needs to be recreated.