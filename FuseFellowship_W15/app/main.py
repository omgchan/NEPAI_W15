import os
from typing import List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# Load environment variables
load_dotenv()


# Structured response schema
class AssistantResponse(BaseModel):
    answer: str = Field(
        description="Clear answer to the user's question"
    )

    confidence: str = Field(
        description="Confidence level: low, medium, or high"
    )

    sources: List[str] = Field(
        description="List of sources used to answer the question"
    )


# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")


# Create LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# Force structured output
structured_llm = llm.with_structured_output(
    AssistantResponse
)


# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant specializing in Nepalese legal documents.

        Answer clearly and concisely.
        Do not invent legal information.
        If you do not know the answer, clearly say that you do not know.
        """
    ),
    ("human", "{question}")
])


# Create chain
chain = prompt | structured_llm


# Run chain
response = chain.invoke({
    "question": "What is Retrieval-Augmented Generation?"
})


print(response)