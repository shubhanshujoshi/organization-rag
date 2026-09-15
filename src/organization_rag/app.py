from __future__ import annotations

from pathlib import Path

import streamlit as st

from organization_rag.document_loader import discover_documents
from organization_rag.llamaindex_rag import LlamaIndexRAG
from organization_rag.upload_ingest import ingest_uploaded_files


st.set_page_config(
    page_title="Organization RAG Assistant",
    page_icon="📚",
    layout="centered",
)


@st.cache_resource
def load_rag():
    return LlamaIndexRAG()


rag = load_rag()


st.title("📚 Organization RAG Assistant")

st.write(
    "Upload departmental documents, index them, and ask questions "
    "using department-authorized retrieval."
)


# ============================================================
# 1. ADD DOCUMENTS
# ============================================================

st.header("1. Add documents")

department = st.selectbox(
    "Department",
    [
        "placement",
        "hr",
        "finance",
        "academic",
        "admissions",
        "administration",
        "library",
        "hostel",
        "it",
        "general",
    ],
)


uploaded_files = st.file_uploader(
    "Drag and drop one or more documents",
    type=[
        "pdf",
        "docx",
        "xlsx",
        "xls",
        "csv",
        "txt",
        "md",
    ],
    accept_multiple_files=True,
    help=(
        "Supported formats: PDF, DOCX, XLSX, XLS, CSV, TXT and Markdown."
    ),
)


if uploaded_files:
    st.write("Selected files:")

    for uploaded_file in uploaded_files:
        st.write(f"• {uploaded_file.name}")

    if st.button(
        "Process and Index Documents",
        type="primary",
    ):
        progress_box = st.empty()

        def show_progress(message: str) -> None:
            progress_box.info(message)

        try:
            with st.spinner(
                "Parsing, chunking, embedding and indexing..."
            ):
                stats = ingest_uploaded_files(
                    uploaded_files=uploaded_files,
                    department=department,
                    progress_callback=show_progress,
                )

            progress_box.success(
                "Documents processed successfully."
            )

            st.success(
                f"Indexed {stats['documents']} document(s), "
                f"{stats['logical_pages']} logical page/sheet(s), "
                f"{stats['chunks']} chunks."
            )

            # Recreate the cached RAG object so it sees
            # the rebuilt Qdrant collection.
            load_rag.clear()
            rag = load_rag()

        except Exception as exc:
            st.error(
                f"Document processing failed: {exc}"
            )


# ============================================================
# 2. ASK A QUESTION
# ============================================================

st.header("2. Ask a question")

question = st.text_input(
    "Question",
    placeholder=(
        "e.g. How many placement attempts are allowed?"
    ),
)


if st.button("Ask"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Searching authorized documents..."
        ):
            try:

                # LlamaIndexRAG.answer() returns a dictionary:
                #
                # {
                #     "answer": "...",
                #     "sources": [...]
                # }
                #
                result = rag.answer(
                    question=question,
                    department=department,
                )

                answer = result["answer"]
                sources = result["sources"]

            except Exception as exc:
                st.error(
                    f"RAG query failed: {exc}"
                )
                st.stop()

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.subheader("Answer")

        st.markdown(answer)

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if sources:

            st.subheader("Sources")

            for source in sources:

                page = source.get("page")
                section = source.get("section")
                source_name = source.get(
                    "source",
                    "Unknown source",
                )
                rank = source.get(
                    "rank",
                    "?",
                )
                score = source.get(
                    "score"
                )

                details = [source_name]

                if page is not None:
                    details.append(
                        f"Page/Sheet {page}"
                    )

                if section:
                    details.append(
                        f"Section: {section}"
                    )

                if score is not None:
                    score_text = f"{score:.3f}"
                else:
                    score_text = "N/A"

                st.write(
                    f"**[{rank}]** "
                    f"{' | '.join(details)} "
                    f"(score: {score_text})"
                )

        else:

            st.info(
                "No authorized sources were retrieved."
            )


# ============================================================
# EXISTING INDEXED DOCUMENTS
# ============================================================

st.divider()

st.caption(
    "Documents are organized under "
    "data/raw/<department>/ and processed into "
    "Markdown before indexing."
)


existing_documents = discover_documents(
    Path("data/raw")
)


if existing_documents:

    with st.expander(
        "Indexed source files"
    ):

        for path in existing_documents:
            st.write(str(path))