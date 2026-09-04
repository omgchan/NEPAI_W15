from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_PATH = Path("data/documents")
DB_PATH = "chroma_db"


def load_documents():
    all_documents = []

    pdf_files = list(DATA_PATH.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()

        # Add the source filename to metadata
        for document in documents:
            document.metadata["source_file"] = pdf_file.name

        all_documents.extend(documents)

    print(f"Total pages/documents loaded: {len(all_documents)}")

    return all_documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks



def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )

    return vector_store
def load_vector_store():
    print("\n--- Loading Vector Store ---")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    return vector_store

def retrieve_documents(vector_store, query, k=4):
    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results

def format_documents(documents):
    formatted_context = []

    for document in documents:
        source = document.metadata.get("source_file", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_context.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{document.page_content}"
        )

    return "\n\n---\n\n".join(formatted_context)

if __name__ == "__main__":
    vector_store = load_vector_store()

    query = "What are the fundamental rights provided by the Constitution of Nepal?"

    results = retrieve_documents(
        vector_store,
        query,
        k=4
    )

    print("\n--- Retrieved Documents ---")

    for i, document in enumerate(results, start=1):
        print(f"\n{'=' * 60}")
        print(f"Result {i}")
        print(f"{'=' * 60}")

        print("\nSource:")
        print(document.metadata.get("source_file"))

        print("\nPage:")
        print(document.metadata.get("page"))

        print("\nContent:")
        print(document.page_content)