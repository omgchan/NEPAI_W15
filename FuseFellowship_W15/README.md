# Nepal Legal AI Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant designed to answer questions using Nepalese legal documents.

The project combines **Large Language Models (LLMs)** with document retrieval to generate answers grounded in the contents of legal documents rather than relying only on the model's pretrained knowledge.

Currently, the system supports:

* Groq LLM integration
* Prompt engineering
* Structured responses using Pydantic
* PDF document ingestion
* Multi-document processing
* Text extraction
* Metadata preservation
* Recursive text chunking

The project is currently being extended with vector search, retrieval, tool calling, a web interface, and containerization.

---

## Overview

Traditional LLM applications answer questions primarily using information learned during model training:

```text
User Question
      ↓
     LLM
      ↓
   Answer
```

This can result in outdated or unsupported answers.

This project uses **Retrieval-Augmented Generation (RAG)**:

```text
User Question
      ↓
Retrieve Relevant Legal Information
      ↓
Question + Retrieved Context
      ↓
     LLM
      ↓
Grounded Answer
```

The assistant retrieves relevant information from legal documents and provides it as context to the LLM before generating a response.

---

## Documents

The current knowledge base contains:

| Document              |     Pages |
| --------------------- | --------: |
| Constitution of Nepal |       202 |
| Muluki Ain            |       905 |
| **Total**             | **1,107** |



## Architecture

The planned architecture of the application is:

```text
                         ┌─────────────────────────┐
                         │     Legal Documents     │
                         │                         │
                         │  Constitution of Nepal  │
                         │       Muluki Ain        │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     PDF Loader          │
                         │      PyPDFLoader        │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     Text Chunking       │
                         │ RecursiveTextSplitter   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      Embeddings         │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       ChromaDB          │
                         │    Vector Database      │
                         └────────────┬────────────┘
                                      │
                                      ▼
User Question ───────────────────► Retrieval
                                      │
                                      ▼
                              Relevant Context
                                      │
                                      ▼
                              Prompt Template
                                      │
                                      ▼
                               Groq LLM
                                      │
                                      ▼
                          Structured Response
```

---

## Technology Stack

| Technology                     | Purpose                      |
| ------------------------------ | ---------------------------- |
| Python                         | Application development      |
| LangChain                      | LLM and RAG orchestration    |
| Groq                           | LLM inference                |
| `openai/gpt-oss-120b`          | Language model               |
| Pydantic                       | Structured output validation |
| PyPDFLoader                    | PDF document ingestion       |
| RecursiveCharacterTextSplitter | Document chunking            |
| Sentence Transformers          | Text embeddings              |
| ChromaDB                       | Vector database              |
| Streamlit                      | Web interface                |
| Docker                         | Containerization             |

---

## Project Structure

```text
FuseFellowship_W15/
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
│
├── app/
│   ├── main.py
│   ├── rag.py
│   └── test_pdf.py
│
├── data/
│   └── documents/
│       ├── Constitution of Nepal.pdf
│       └── Muluki Ain.pdf
│
└── chroma_db/
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd FuseFellowship_W15
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Do not commit the `.env` file to version control.

An example configuration is provided in `.env.example`.

---

## LLM Integration

The application uses the Groq API through LangChain.

Current model configuration:

```python
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)
```

The LLM can be invoked directly:

```text
User Question
      ↓
Groq API
      ↓
LLM
      ↓
Response
```

---

## Prompt Engineering

The assistant uses `ChatPromptTemplate` to separate system instructions from user input.

The system prompt defines the assistant's behavior, including:

* Acting as an assistant focused on Nepalese legal documents.
* Providing clear and concise answers.
* Avoiding unsupported or fabricated legal information.
* Acknowledging when sufficient information is unavailable.

The prompt and LLM are connected using LangChain's pipeline operator:

```python
chain = prompt | llm
```

The chain is executed using:

```python
response = chain.invoke({
    "question": "What is RAG?"
})
```

---

## Structured Output

The project uses Pydantic to define a predictable response format.

```python
class AssistantResponse(BaseModel):
    answer: str
    confidence: str
    sources: List[str]
```

The LLM is configured to generate output matching this structure:

```python
structured_llm = llm.with_structured_output(
    AssistantResponse
)
```

This allows responses to be accessed programmatically:

```python
response.answer
response.confidence
response.sources
```

Structured output is useful for integrating AI responses into web applications because the output format is predictable.

---

## Document Ingestion

PDF documents are loaded using `PyPDFLoader`.

Each page is converted into a LangChain `Document` object.

Current ingestion results:

```text
2 PDF Documents
      ↓
1,107 Pages
      ↓
1,107 Document Objects
```

Metadata from the source documents is preserved.

Example metadata:

```python
{
    "source_file": "Constitution of Nepal.pdf",
    "page": 0,
    "page_label": "1"
}
```

This metadata can later be used to provide source references with generated answers.

---

## Text Chunking

Large legal documents are divided into smaller chunks before embedding.

The application uses:

```python
RecursiveCharacterTextSplitter
```

Configuration:

```python
chunk_size=1000
chunk_overlap=200
```

Current results:

```text
1,107 Document Pages
        ↓
Recursive Text Splitting
        ↓
4,147 Text Chunks
```

Chunk overlap preserves context between adjacent chunks when information spans chunk boundaries.

---

## Running the Project

### Test PDF Loading

```bash
python app/test_pdf.py
```

### Run Document Chunking

```bash
python app/rag.py
```

### Run the LLM Application

```bash
python app/main.py
```

---

## RAG Pipeline

The complete RAG pipeline will have two phases.

### 1. Indexing Phase

This phase prepares the documents for retrieval.

```text
PDF Documents
      ↓
Text Extraction
      ↓
Document Chunking
      ↓
Embedding Generation
      ↓
ChromaDB Storage
```

### 2. Question Answering Phase

```text
User Question
      ↓
Query Embedding
      ↓
Vector Similarity Search
      ↓
Relevant Document Chunks
      ↓
Question + Context
      ↓
Groq LLM
      ↓
Structured Answer
```

---

## Why RAG?

LLMs can generate convincing answers even when the information is unsupported.

RAG reduces this problem by retrieving relevant information from an external knowledge base before generating the response.

```text
Without RAG:

Question → LLM → Answer


With RAG:

Question
   ↓
Retrieve Relevant Documents
   ↓
Context
   ↓
LLM
   ↓
Grounded Answer
```

For a legal assistant, grounding answers in source documents is particularly important.

---

## Roadmap

### Phase 1 — LLM Foundation

* [x] Groq integration
* [x] Prompt engineering
* [x] Structured output

### Phase 2 — Document Processing

* [x] PDF ingestion
* [x] Multi-document support
* [x] Text extraction
* [x] Metadata preservation
* [x] Text chunking

### Phase 3 — RAG

* [ ] Generate embeddings
* [ ] Create ChromaDB vector store
* [ ] Implement semantic retrieval
* [ ] Connect retrieval to the LLM
* [ ] Return source information

### Phase 4 — Application

* [ ] Streamlit user interface
* [ ] Tool calling
* [ ] Error handling
* [ ] Retry mechanism
* [ ] Caching
* [ ] Model fallback

### Phase 5 — Deployment

* [ ] Dockerfile
* [ ] Docker Compose
* [ ] Deployment documentation

---

## Disclaimer

This project is developed for educational and research purposes.

The responses generated by this system should not be considered official legal interpretations or professional legal advice. Users should consult qualified legal professionals or official legal sources when making legal decisions.
