import streamlit as st

from organization_rag.rag import OrganizationRAG


st.set_page_config(
    page_title="Organization RAG Assistant",
    page_icon="📚",
    layout="centered",
)


@st.cache_resource
def load_rag():
    return OrganizationRAG()


rag = load_rag()

st.title("📚 Organization RAG Assistant")
st.write(
    "Ask questions about the organization's documents. "
    "Answers are generated only from the retrieved documents."
)

question = st.text_input(
    "Ask a question",
    placeholder="e.g. How many placement attempts are allowed?",
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the documents..."):
            answer = rag.answer(question)

        st.subheader("Answer")
        st.markdown(answer)