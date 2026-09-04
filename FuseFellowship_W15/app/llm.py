import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from rag import load_vector_store, retrieve_documents, format_documents

load_dotenv()

# --- 1. Structured Response Schema ---
class AssistantResponse(BaseModel):
    answer: str = Field(
        description="Clear answer to the user's question"
        )

    confidence: str = Field(
        description="Confidence level: low, medium, or high"
        )
        
    sources: List[str] = Field(
        description="List of sources/articles used to answer the question"
        )

# --- 2. Tool Calling Definition ---
@tool
def calculate_fine_or_duration(base_unit: float, multiplier: float) -> str:
    """Calculates legal fines or sentence adjustments based on statutory multipliers."""
    result = base_unit * multiplier
    return f"Calculated value: {result}"

tools = [calculate_fine_or_duration]

# --- 3. Primary & Fallback LLM Setup with Tool Binding ---
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")


primary_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    groq_api_key=api_key
)

fallback_llm = ChatGroq(
    model="meta-llama/llama-prompt-guard-2-22m",
    temperature=0,
    groq_api_key=api_key
)

# Bind tools and fallback provider
llm_with_tools = primary_llm.bind_tools(tools).with_fallbacks([fallback_llm.bind_tools(tools)])
structured_llm = llm_with_tools.with_structured_output(AssistantResponse)

# --- 4. Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant specializing in Nepalese legal documents.
        Answer the user's question using ONLY the provided legal context.
        Do not invent legal information.
        If the provided context does not contain enough information,
        clearly state that the answer cannot be determined from the provided documents.
        Answer clearly and concisely.
        """
    ),
    (
        "human",
        """
        Use the following legal context to answer the question.

        LEGAL CONTEXT:
        {context}

        QUESTION:
        {question}
        """
    )
])

chain = prompt | structured_llm

# --- 5. Reliable Execution Wrapper ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
async def ask_legal_assistant_async(question: str, context: str) -> AssistantResponse:
    """Asynchronously calls the chain with built-in retry logic."""
    return await chain.ainvoke({"question": question, "context": context})

vector_store = load_vector_store()