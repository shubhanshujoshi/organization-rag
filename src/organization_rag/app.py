import streamlit as st

from organization_rag.llamaindex_rag import LlamaIndexRAG


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
    "Ask questions about organization documents. "
    "Your department determines which documents can be retrieved."
)


department = st.selectbox(
    "Select your department",
    [
        "placement",
        "hr",
        "finance",
        "academic",
    ],
)


question = st.text_input(
    "Ask a question",
    placeholder="e.g. How many placement attempts are allowed?",
)


if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching authorized documents..."):

            answer, sources = rag.answer(
                question=question,
                department=department,
            )

        st.subheader("Answer")
        st.markdown(answer)

        if sources:
            st.subheader("Sources")

            for source in sources:
                st.write(
                    f"**[{source['rank']}]** "
                    f"{source['source']} "
                    f"(score: {source['score']:.3f})"
                )