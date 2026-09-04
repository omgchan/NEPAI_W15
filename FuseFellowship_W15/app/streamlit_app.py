import asyncio
import streamlit as st

from rag import (
    load_vector_store,
    retrieve_documents,
    format_documents
)
from llm import ask_legal_assistant_async

# Page Configuration
st.set_page_config(
    page_title="Nepal Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Load Vector Store
@st.cache_resource
def get_vector_store():
    return load_vector_store()

vector_store = get_vector_store()

# UI Setup
st.title("⚖️ Nepal Legal AI Assistant")

st.write(
    "Ask questions based on the Constitution of Nepal and the historical Muluki Ain (1854)."
)

st.warning(
    "Note: The Muluki Ain included in this knowledge base is a historical legal document and may not represent current Nepali law. "
    "Disclaimer: This application is for educational purposes only and does not provide professional legal advice."
)

question = st.text_input(
    "Ask a question about the legal documents:",
    placeholder="Example: What does the Constitution say about the right to equality?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching legal documents & generating answer..."):
            # 1. Retrieve documents
            retrieved_documents = retrieve_documents(
                vector_store,
                question,
                k=8
            )

            # 2. Extract verified document sources
            retrieved_sources = list({
                f"{doc.metadata.get('source_file', 'Document')}, "
                f"Page {doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))}"
                for doc in retrieved_documents
            })

            # 3. Format context
            context = format_documents(retrieved_documents)

            # 4. Asynchronous LLM Execution
            response = asyncio.run(
                ask_legal_assistant_async(question=question, context=context)
            )

        # Display Response
        st.subheader("Answer")
        st.write(response.answer)

        # Display Confidence Badge
        st.subheader("Confidence")
        confidence_color = {
            "high": "High",
            "medium": "Medium",
            "low": "Low"
        }.get(response.confidence.lower(), f"{response.confidence}")
        st.markdown(f"**{confidence_color}**")

        # Display Citations & Sources
        st.subheader("Sources & Citations")
        
        # Priority to model-extracted sources, falling back to vector retrieval metadata
        display_sources = response.sources if response.sources else retrieved_sources
        for src in display_sources:
            st.markdown(f"* {src}")