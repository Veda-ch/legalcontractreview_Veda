"""File-upload test UI for the legal contract review pipeline (stage 7 scaffold).

Run:  streamlit run app.py
"""
import os
import streamlit as st
from pipeline import run

UPLOAD_DIR = "data/uploads"

st.title("Legal Contract Review")

uploaded = st.file_uploader("Upload a contract", type=["pdf", "docx"])

if uploaded is None:
    st.info("Upload a .pdf or .docx contract above to analyze it.")
else:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        with st.spinner("Analyzing contract..."):
            doc, report = run(save_path)
    except Exception as e:
        st.error(str(e))
    else:
        clauses_by_id = {c.clause_id: c for c in doc.clauses}

        st.write(f"**Detected source language:** `{doc.source_language}`")
        st.metric("Overall risk", report.overall_risk.upper(), f"score {report.overall_score}")
        st.write(report.summary)

        for finding in report.clause_findings:
            clause = clauses_by_id.get(finding.clause_id)
            flagged = finding.clause_id in report.flagged_for_review
            heading = clause.heading if clause and clause.heading else finding.clause_id
            title = f"{'🚩 FLAGGED — ' if flagged else ''}{heading} ({finding.risk_level} risk)"

            with st.expander(title):
                if clause:
                    st.write("**Text (original):**")
                    st.write(clause.text_original)
                    st.write("**Text (English):**")
                    st.write(clause.text_en)

                st.write("**Risk level:**", finding.risk_level)
                st.write("**Review decision:**", finding.review_decision, "—", finding.review_reason)

                if finding.issues:
                    st.write("**Issues:**")
                    for issue in finding.issues:
                        st.write(f"- [{issue.severity}] {issue.category}: {issue.description}")
                else:
                    st.write("**Issues:** none")

        with st.expander("Raw JSON"):
            st.json(report.model_dump())
