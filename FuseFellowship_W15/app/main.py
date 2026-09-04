from rag import (
    load_vector_store,
    retrieve_documents,
    format_documents
)

from llm import chain


if __name__ == "__main__":

    vector_store = load_vector_store()

    question = "What does the Constitution of Nepal say about the right to equality?"

    retrieved_documents = retrieve_documents(
        vector_store,
        question,
        k=8
    )

    sources = list({
        f"{doc.metadata.get('source_file')}, "
        f"Page {doc.metadata.get('page_label', doc.metadata.get('page'))}"
        for doc in retrieved_documents
    })

    context = format_documents(retrieved_documents)

    response = chain.invoke({
        "question": question,
        "context": context
    })

    print("\n--- Answer ---")
    print(response.answer)

    print("\n--- Confidence ---")
    print(response.confidence)

    print("\n--- Sources ---")
    print(sources)