# Legal Contract Review — Current Project Status

## 1. Project Flow

```text
Contract (PDF/DOCX)
        │
        ▼
┌─────────────────────┐
│ 1. Document Parsing │
│ PDF/DOCX → Clauses  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. Classification   │
│ LegalBERT + LEDGAR  │
│      100 classes    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. Retrieval        │
│ FAISS + GraphRAG   │
│ Legal KB            │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 4. Risk Analysis    │
│ Evidence-based      │
│ multi-factor review │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 5. Uncertainty      │
│ RL Agent            │
│ (not completed yet) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 6. Report / UI      │
└─────────────────────┘

2. Current Progress

| Stage                 | Status         | Main Technology                |
| --------------------- | -------------- | ------------------------------ |
| Document Parsing      | ✅ Done         | PDF/DOCX parsing               |
| Clause Classification | ✅ Done         | LegalBERT + LEDGAR             |
| Knowledge Retrieval   | ✅ Done         | FAISS + GraphRAG               |
| Risk Analysis         | ✅ Done         | Multi-factor analysis + Ollama |
| RL Uncertainty        | 🔄 Pending     | RL model/performed only based on confidence score for now|
| Final Report          | 🔄 Integration | Pipeline + UI/basic report in json format implemented|
| Agentic Orchestration | 🔄 Pending     | LangGraph / Agents             |

3. What Each Completed Stage Does

  1. Document Parsing ✅
PDF / DOCX
   ↓
Extract text
   ↓
Detect / segment clauses
   ↓
Original + parsed clause text
The parsed JSON is retained and passed to the next stages.

  2. Clause Classification ✅

Model: Fine-tuned LegalBERT

Training data: LEDGAR

Classes: 100 original LEDGAR categories
Clause
  ↓
LegalBERT tokenizer
  ↓
Fine-tuned LegalBERT
  ↓
100-class probabilities
  ↓
Predicted clause type
  +
Confidence
  +
Top-3 predictions
  +
Attention-based salient terms

Example:

Clause → Indemnifications

Confidence: 0.6899

Top predictions:
1. Indemnifications
2. Indemnity
3. Releases

  3. Knowledge Retrieval ✅

Knowledge Base:

LEDGAR — 27,328 records
CUAD — 3,000 records
MAUD — 3,000 records
Total — 33,328 records
Legal Knowledge Base
        │
        ├── FAISS
        │     └── similarity-based retrieval
        │
        └── GraphRAG
              └── category / relationship-based
                  candidate selection

FAISS retrieves semantically similar legal records.

GraphRAG narrows/selects relevant records using the lightweight legal knowledge graph.

The retrieved records are then provided as evidence/context for risk analysis.

  4. Risk Analysis ✅

Risk analysis does not assign an arbitrary score out of 100.

Instead, it examines multiple evidence-based factors.

Clause
  │
  ├── Classification result
  ├── Classification confidence
  ├── Retrieval relevance
  ├── Retrieved legal evidence
  ├── Original vs extracted text
  ├── Text differences / missing information
  └── Supported legal risk factors
          │
          ▼
     Risk Analysis
          │
          ▼
 Evidence + Reasoning
          │
          ▼
 Restricted Ollama explanation

Ollama is restricted to explaining the provided evidence and analysis. It should not invent legal facts, clauses, evidence, or risk factors.

4. Uncertainty Agent 🔄
Current baseline

At present, uncertainty is based on the classifier confidence.

LegalBERT confidence
        ↓
Confidence threshold
        ↓
Low-confidence flag
Planned version

This will be replaced/extended by the RL-based uncertainty agent.

Classification
      +
Retrieval
      +
Risk analysis
      +
Other uncertainty signals
        ↓
      RL Agent
        ↓
Decision / uncertainty assessment

5. Agentic AI — Next Major Integration

The current pipeline contains the individual AI components.

The next step is to connect them through an agentic orchestration layer.

                  Supervisor Agent
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Parsing Agent   Classification   Retrieval Agent
                        Agent             │
        │               │                │
        └───────────────┼────────────────┘
                        ▼
                  Risk Analysis
                      Agent
                        │
                        ▼
                 Uncertainty Agent
                        │
                        ▼
                  Report Agent

The Supervisor Agent will determine the workflow and coordinate the specialized agents rather than simply running every component independently.

6. What Is Left
🔄 Remaining Work
Complete the RL-based uncertainty agent.
Integrate the uncertainty agent into the pipeline.
Integrate the agentic/Supervisor architecture.
Connect the final report generation properly.
Final UI improvements and end-to-end testing.
Test the complete system on multiple contracts.

7. Important Current State

Completed:

✅ Parsing
✅ LegalBERT classification
✅ 100 LEDGAR classes
✅ 33,328-record legal knowledge base
✅ FAISS retrieval
✅ GraphRAG retrieval
✅ Evidence-based risk analysis
✅ Restricted Ollama explanation

Not yet completed:

🔄 RL uncertainty agent
🔄 Agentic orchestration
🔄 Final end-to-end integration

## 8. How to Run the Current System

### First-time setup

```bash
pip install -r requirements.txt

Make sure the following are present:

legalbert_finetuned/ — trained 100-class LegalBERT
data/knowledge_base/legal_records.jsonl — unified legal KB
FAISS index files — generated using the FAISS index builder
Ollama + required model — for restricted risk-analysis explanations

Start Ollama

If Ollama is already running, do not run ollama serve again.

Check:

ollama list

Run the application

streamlit run app.py