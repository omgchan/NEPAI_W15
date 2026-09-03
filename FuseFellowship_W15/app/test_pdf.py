from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path


DATA_PATH = Path("data/documents")
pdf_files = list(DATA_PATH.glob("*.pdf"))


print(f"Found {len(pdf_files)} PDF files in {DATA_PATH}.")

for pdf_file in pdf_files:
    print("-" * 60)
    print(f"Loading PDF file: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from {pdf_file.name}.")
    print(f"\nFist Page Content")
    print(documents[0].page_content[:500])
    print("\n")