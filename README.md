# AI Document Assistant

A locally running Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their contents. It uses Streamlit, ChromaDB, and locally hosted language models through Ollama to generate document-grounded answers.

## Features

- **PDF processing:** Extracts text from PDF documents and divides it into overlapping chunks.
- **Semantic search:** Generates embeddings and retrieves relevant document sections using ChromaDB.
- **Context-aware question answering:** Supports conversation context and follow-up questions through question rewriting.
- **Grounded responses:** Uses document context to generate answers with supporting PDF page references.
- **Document reuse:** Identifies previously indexed PDFs using SHA-256-based document identity.
- **Local AI inference:** Uses Ollama models without requiring a hosted LLM API.
- **RAG evaluation dashboard:** Provides test-case creation and automated retrieval and keyword evaluation.
- **Docker deployment:** Runs the Streamlit application and ChromaDB using Docker Compose.

## Architecture

The application follows a Retrieval-Augmented Generation pipeline:

1. A user uploads a PDF document.
2. PyPDF extracts text from the document.
3. Extracted text is divided into chunks of 1,000 characters with 200 characters of overlap.
4. Ollama generates embeddings using `nomic-embed-text`.
5. ChromaDB stores the embeddings, text chunks, and page metadata.
6. A user submits a question through the chat interface.
7. The `llama3.2` model uses recent conversation history to rewrite follow-up questions as standalone questions.
8. The application retrieves up to five relevant chunks using the standalone question.
9. Retrieved context is passed to the answer-generation model.
10. The model generates a document-grounded answer with supporting page references.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Programming language | Python |
| PDF extraction | pypdf |
| Vector database | ChromaDB |
| Embedding model | nomic-embed-text |
| Question rewriting model | llama3.2 |
| Answer generation model | granite4.2:8b |
| Local model runtime | Ollama |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Testing | pytest |
| CI/CD | GitHub Actions |

## System Requirements

Install the following before running the application:

- Docker Desktop
- Ollama
- Git

The Streamlit application and ChromaDB run in Docker containers. Ollama runs locally on the host machine.

Internet access is required initially to install dependencies and download the models. Once the required software and models are installed, document processing and inference can run locally without a hosted LLM API.

## Model Setup

Download the required Ollama models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
ollama pull granite4.2:8b
```

Verify the installed models:

```bash
ollama list
```

Make sure Ollama is running before starting the application.

## Installation and Execution

Clone the repository:

```bash
git clone https://github.com/Mosesisaac12/ai-document-assistant.git
cd ai-document-assistant
```

Build and start the application:

```bash
docker compose up --build -d
```

Open the main application:

**http://localhost:8501**

The application connects to Ollama through `host.docker.internal:11434` and connects to ChromaDB through the internal Docker network.

The provided Compose configuration is intended for Docker Desktop environments. Linux deployments may require additional host networking configuration.

## Using the Application

1. Open the application in your browser.
2. Upload a text-based PDF document.
3. Wait for document indexing to finish.
4. Enter a question about the uploaded document.
5. Review the generated answer and its supporting page references.
6. Ask follow-up questions using the active conversation context.

The application reuses an existing indexed collection when the same PDF and indexing configuration are detected.

**Note:** Scanned or image-only PDFs without extractable text are not supported by the current PDF extraction pipeline.

## RAG Evaluation Dashboard

The project includes a separate evaluation dashboard.

To open it:

1. Open the main application.
2. Expand **Developer Options** in the sidebar.
3. Select **Start Evaluation Dashboard**.
4. Open the dashboard at **http://localhost:8502**.
5. Upload the evaluation PDF.
6. Create test cases using a question, expected answer, expected PDF pages, and expected keywords.
7. Run the evaluation and inspect the results.

### Evaluation Metrics

**Retrieval Pass**

Checks whether all expected PDF pages appear in the retrieved results.

**Page Recall**

Measures the proportion of expected pages retrieved:

`Page Recall = Matched Expected Pages / Total Expected Pages`

**Keyword Coverage**

Measures the proportion of expected keywords or phrases found in the generated answer:

`Keyword Coverage = Matched Keywords / Total Expected Keywords`

**Automatic Pass**

A test case passes automatically when retrieval passes and keyword coverage reaches 100%.

**Average Response Time**

Measures the average time taken to execute the RAG question-answering pipeline across the test cases.

The evaluation uses deterministic page and keyword metrics. It does not currently perform semantic answer comparison or LLM-as-a-judge evaluation. Expected answers are retained for reference.

Test cases and evaluation results are maintained in Streamlit session state.

## Running Automated Tests

Install the Python dependencies in a local virtual environment:

```bash
python -m pip install -r requirements.txt
```

Run the unit tests:

```bash
python -m pytest tests/ -v
```

The current test suite validates the evaluation logic. It does not represent a complete end-to-end integration test of Ollama, ChromaDB, and the Streamlit application.

GitHub Actions automatically runs the test suite on pushes and pull requests.

## Project Structure

```text
ai-document-assistant/
├── .github/
│   └── workflows/
│       └── tests.yml
├── tests/
│   └── test_evaluation.py
├── rag_app.py
├── rag_engine.py
├── evaluation.py
├── evaluation_app.py
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## Data Persistence

ChromaDB stores indexed document data in a named Docker volume:

`ai-chatbot_chroma_data`

This allows indexed documents to remain available across container restarts.

To stop the application:

```bash
docker compose down
```

This preserves the named ChromaDB volume.

**Warning:** Avoid `docker compose down -v` if you want to retain your indexed documents, because it removes the Compose-managed volumes.

## Current Limitations

- The current PDF extraction pipeline supports text-based PDFs, not scanned PDFs requiring OCR.
- Chat history is maintained in the active Streamlit session and is not persisted across sessions.
- Evaluation test cases are session-based rather than permanently stored.
- Automated evaluation uses page retrieval and keyword coverage, not semantic answer-quality scoring.
- The current automated test suite focuses on evaluation logic rather than full application integration.

## License

No license has been specified yet.