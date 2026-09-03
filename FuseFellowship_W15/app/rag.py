from pathlib import Path


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_PATH = Path("data/documents")


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


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print("\n--- First Chunk ---\n")
    print(chunks[0].page_content)

    print("\n--- Metadata ---")
    print(chunks[0].metadata)