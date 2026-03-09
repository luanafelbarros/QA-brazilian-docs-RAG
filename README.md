# Project 1 - RAG: Retrieval Augmented Generation System

## Students
* Luana Felipe de Barros (luanafelbarros@gmail.com)

## Project Overview
This project implements a Retrieval Augmented Generation (RAG) system using a vector database (Pinecone) to store document embeddings and a Large Language Model (LLM) from Groq to generate contextual responses. The objective is to create a system capable of answering questions based on a provided set of documents, evaluating the relevance and accuracy of the generated responses.

## Project Structure
The project is organized into the following stages:
1.  **Data Ingestion and Preparation**: Loading, cleaning, and formatting documents.
2.  **Embedding Model Definition**: Choosing and utilizing an embedding model to transform text into numerical vectors.
3.  **Vector Database Configuration (Pinecone)**: Creating and indexing documents in Pinecone.
4.  **Document Retrieval**: Implementing logic to search for relevant documents in Pinecone.
5.  **Response Generation with LLM**: Using a Groq model (LLaMA) to generate responses based on retrieved documents.
6.  **Response Evaluation**: Analyzing the quality of responses using RAGAS and RAG Evaluator metrics.

## Setup and Installation

### Prerequisites
*   Pinecone account with API key.
*   Groq account with API key.
*   Google Colab access (or similar Python environment).

### 1. Clone the Repository and Install Dependencies
```bash
# Ensure Colab is configured correctly or adjust paths as needed.
# This project uses files located in a mounted Google Drive.
```

Python dependencies are installed using pip:
```python
!pip install -qU datasets==2.14.5 groq pinecone-client==4.1.0 python-dotenv
!pip install -qU cohere semantic_router
!pip install sentence-transformers
!pip -q install -qU "langchain[groq]"
!pip install rapidfuzz
!pip install rouge_score -q
!pip install sacrebleu -q
!pip install ragas
!pip install rag-evaluator
```

### 2. Configure Environment Variables
Create a `.env` file (or use Colab's secrets manager) with the following keys:
```
PINECONE="your_pinecone_api_key"
GROQ_API_KEY="your_groq_api_key"
```

## Running the Project

### 1. Data Loading and Preparation
Data is loaded from a `documents.csv` file, and an `id` is added to each document. Null values are removed, and metadata is formatted for use with Pinecone.

### 2. Embedding Model Definition
The multilingual embedding model `paraphrase-multilingual-MiniLM-L12-v2` from Sentence Transformers was used, which proved more effective for Portuguese compared to the `dwzhu/e5-base-4k` initially tested.

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
```

### 3. Pinecone Configuration and Indexing
Connection to Pinecone is established, and an index (`inf0084multi`) is created. Documents are then inserted in batches into the index.

```python
from pinecone import Pinecone, ServerlessSpec
import time

api_key = os.environ.get("PINECONE") # Or use getpass for manual entry
pc = Pinecone(api_key=api_key)

index_name = "inf0084multi"
# ... (code for index creation and connection)

# ... (code for upserting embeddings)
```

### 4. Document Retrieval
The `get_docs` function is used to query the Pinecone index and retrieve the `k` most relevant documents for a given query.

```python
def get_docs(query: str, top_k: int) -> list[str]:
    # ... (code to encode the query and search Pinecone)
```

### 5. Response Generation with Groq
A LLaMA model (via `langchain-groq`) is used to generate responses based on retrieved documents. A specific prompt is constructed to ensure concise and context-grounded answers.

```python
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = init_chat_model("llama3-70b-8192", model_provider="groq", temperature=0)

template = """You are a specialized assistant for answering questions.
# ... (RAG prompt and pipeline)

def rag_pipeline(query: str):
    # ... (code for retrieval and response generation)
```

### 6. Response Evaluation
Responses are evaluated using metrics from the RAGAS (without LLM usage) and RAG Evaluator libraries, focusing on semantic similarity, BLEU and ROUGE scores, and BERTScore. A dataset of question-answer pairs (`qa_pairs.csv`) is used as a reference.

#### Metrics Used:
*   **NonLLMStringSimilarity** (RAGAS)
*   **StringPresence** (RAGAS)
*   **ExactMatch** (RAGAS)
*   **SemanticSimilarity** (RAGAS)
*   **RougeScore** (RAGAS)
*   **BleuScore** (RAGAS)
*   **BLEU** (RAG Evaluator)
*   **ROUGE-1** (RAG Evaluator)
*   **BERT P, R, F1** (RAG Evaluator)
*   **Perplexity** (RAG Evaluator)
*   **Diversity** (RAG Evaluator)
*   **Racial Bias** (RAG Evaluator)

#### Evaluation Results:
The metrics indicated high semantic similarity (BERTScore ~82%), demonstrating that the RAG pipeline successfully generated responses with meanings close to the references. Metrics like BLEU were lower due to their sensitivity to word variations and structure, but the overall evaluation was positive.

## Observations and Comments
*   **Embedding Model**: The switch to a multilingual model (`paraphrase-multilingual-MiniLM-L12-v2`) was crucial for improving the relevance of similarity scores and retrieval quality.
*   **Prompt Engineering**: A One-shot Learning strategy was implemented in the LLM prompt to encourage concise responses, aligning with the format of the evaluation dataset. This improved evaluation on word-overlapping metrics.
*   **Limitations**: Questions outside the scope of the documents resulted in correct "I don't know the answer" responses. For questions in English, the model had difficulty adhering to the desired concise pattern.
