# AI Research Assistant

A **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about a collection of PDF research documents and receive relevant, context-aware answers.

The application combines **LangChain, Hugging Face embeddings, ChromaDB, and Groq LLMs** with a Streamlit interface.

## Features

* Load and process multiple PDF documents
* Clean noisy and irrelevant document content
* Split documents into meaningful chunks
* Semantic search using vector embeddings
* MMR-based retrieval for diverse relevant chunks
* Generate answers using a Groq-hosted LLM
* Interactive Streamlit question-answer interface
* Fallback to general LLM knowledge when relevant document context is unavailable
* Display retrieved document chunks with source file and page information

## Tech Stack

* **Python**
* **Streamlit** — Web application interface
* **LangChain** — RAG pipeline and LLM orchestration
* **Hugging Face** — Sentence-transformer embeddings
* **ChromaDB** — Vector database
* **Groq** — LLM inference
* **PyPDF** — PDF document loading
* **python-dotenv** — Environment variable management

## Project Structure

```text
ai-research-assitant/
│
├── data/
│   └── *.pdf
│
├── chroma_db/
│   └── Vector database files
│
├── app.py
├── .env
├── requirements.txt
└── README.md
```

## How It Works

```text
PDF Documents
      ↓
Document Loading
      ↓
Document Cleaning
      ↓
Text Chunking
      ↓
Hugging Face Embeddings
      ↓
Chroma Vector Database
      ↓
MMR Retrieval
      ↓
Relevant Context
      ↓
Groq LLM
      ↓
Generated Answer
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ai-research-assitant
```

### 2. Create and activate the virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Add PDF documents

Place the research PDFs you want the application to work with inside:

```text
data/
```

## Run the Application

Activate the virtual environment:

```powershell
venv\Scripts\activate
```

Then start Streamlit:

```powershell
streamlit run app.py
```

The application will open in your browser.

## Example

Enter a question related to the uploaded research documents:

```text
What are the key challenges discussed in the research papers?
```

The application retrieves relevant document chunks and uses them as context for generating the answer.

## RAG Pipeline

The application uses the following RAG process:

1. **Document Loading** — PDF files are loaded using `PyPDFLoader`.
2. **Document Cleaning** — Unnecessary front matter, short pages, and common copyright/legal noise are filtered.
3. **Chunking** — Documents are divided into overlapping chunks using `RecursiveCharacterTextSplitter`.
4. **Embedding** — Chunks are converted into vector representations using `all-MiniLM-L6-v2`.
5. **Vector Storage** — Embeddings are stored in ChromaDB.
6. **Retrieval** — MMR retrieves relevant and diverse document chunks.
7. **Generation** — The retrieved context is passed to the Groq LLM to generate the final response.
8. **Fallback** — If relevant document context cannot be retrieved, the application generates an answer using general technical knowledge.

## Environment Variables

| Variable       | Description                         |
| -------------- | ----------------------------------- |
| `GROQ_API_KEY` | API key used to access the Groq LLM |



