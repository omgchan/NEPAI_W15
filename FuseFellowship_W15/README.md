# ⚖️ Nepal Legal AI Assistant

A production-grade Retrieval-Augmented Generation (RAG) AI assistant designed to answer legal queries grounded in the **Constitution of Nepal** and the historical **Muluki Ain (1854)**.

This project uses modern LLM orchestration, vector search, structured outputs, function calling, reliability backoffs, and containerization.

---

## Architecture

```text
                               ┌─────────────────────────┐
                               │     Legal Documents     │
                               │  Constitution of Nepal  │
                               │       Muluki Ain        │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │  Ingestion & Chunking   │
                               │ PyPDFLoader / Recursive │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │   Chroma Vector DB      │
                               │  Sentence-Transformers  │
                               └────────────┬────────────┘
                                            │
                                            ▼
User Question ───────────────────► Async Vector Retrieval
                                            │
                                            ▼
                                   Retrieved Context
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │ Prompt & Tools Engine   │
                               │  Fine/Sentence Calc     │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │ Groq API (Primary Model)│
                               │ Fallback: Llama-3.3-70b │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │ Streamlit Web Interface │
                               │ (Structured JSON/Badge) │
                               └─────────────────────────┘
```

---

## Architectural Notes & Design Choices

* **vLLM Alternative (Groq API):** Per assignment requirements, local open-source inference via vLLM was specified. However, due to severe hardware and financial constraints for local GPU hosting, the system uses the **Groq API** (`openai/gpt-oss-120b` with `llama-3.3-70b-versatile` fallback). This satisfies open-source model inference constraints at high speed with zero cloud GPU cost.
* **CPU Optimization:** Container builds explicitly isolate CPU-only PyTorch dependencies (`--extra-index-url https://download.pytorch.org/whl/cpu`), preventing bloated multi-gigabyte CUDA installs and ensuring light, fast container execution.

## Features

* **Grounded RAG Pipeline:** Context-constrained prompting eliminates legal hallucination; answers cite exact source documents and pages.
* **Structured Output Validation:** Enforces JSON responses using Pydantic schemas (`answer`, `confidence`, `sources`).
* **Function / Tool Calling:** Native binding for legal fine and sentence multiplier tools.
* **Production Reliability:** Implements retry wrappers with exponential backoff (`tenacity`) and automated provider fallbacks.
* **Async Execution:** Asynchronous processing (`ask_legal_assistant_async`) integrated directly into the Streamlit Web Interface.

---

## Quickstart (Recommended Method: Docker)

The fastest and cleanest way for a TA to run this project is via **Docker Compose**.

### Prerequisites

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* A **Groq API Key** (Free tier available at [console.groq.com](https://console.groq.com/)).

### Step 1: Environment Setup

Create a `.env` file in the project root containing your API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Step 2: Launch Application

Run the following command in the root folder:

```bash
docker-compose up --build
```

### Step 3: Access the Interface

Open your browser and navigate to:

`http://localhost:8501`

---

## Manual Local Setup (Without Docker)

If running directly on a local host system without Docker:

### 1. Clone and Set Up a Virtual Environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the `.env` File

Ensure `GROQ_API_KEY` is present in your root `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Launch the Streamlit Web UI

```bash
streamlit run app/streamlit_app.py
```

---

## Technology Stack

| Technology         | Purpose                                               |
| ------------------ | ----------------------------------------------------- |
| **Python 3.11**    | Core runtime engine                                   |
| **LangChain**      | RAG pipeline orchestration, prompts, and tool binding |
| **Groq API**       | High-throughput LLM inference                         |
| **ChromaDB**       | Local persistent vector storage                       |
| **Pydantic**       | Schema enforcement and structured JSON parsing        |
| **Tenacity**       | Retry and exponential backoff engine                  |
| **Streamlit**      | Frontend web UI                                       |
| **Docker Compose** | Multi-layer container setup                           |

---

## Project Structure

```text
FuseFellowship_W15/
├── app/
│   ├── llm.py            # LangChain, Groq API, Tool binding, Retries & Fallbacks
│   ├── rag.py            # Vector DB ingestion, search retrieval, context formatting
│   └── streamlit_app.py  # Web UI with async execution & confidence badges
│
├── data/
│   └── documents/        # Knowledge base source legal PDFs
│
├── chroma_db/            # Persistent vector database directory
├── Dockerfile            # Optimized lightweight CPU container definition
├── docker-compose.yml    # Service orchestration and volume mounting
├── .dockerignore         # Docker context build exclusion rules
├── requirements.txt      # Dependency manifest (CPU PyTorch optimized)
└── README.md             # System documentation
```

---

## Disclaimer

This application is built strictly for educational and research purposes. It provides context-grounded information from historical and contemporary legal texts but does **not** constitute formal or professional legal advice.
