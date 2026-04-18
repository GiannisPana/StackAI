# StackAI - RAG over PDFs

StackAI is a FastAPI backend for uploading PDF files and answering questions over their contents with cited responses. It uses the Mistral AI API for embeddings, chat, and OCR, and keeps search data locally instead of using a third-party vector database.

## What is included

- PDF ingestion through `POST /ingest`
- Querying through `POST /query`
- A small chat UI served at `/`
- Local storage with SQLite, NumPy, and JSON files under `data/`
- Hybrid retrieval that combines embedding search and keyword search
- Citation-based answers with an "insufficient evidence" refusal path
- Document listing and soft-delete endpoints

## How it works

1. Upload one or more PDF files.
2. Extract text with PyMuPDF and use OCR fallback for low-text pages.
3. Chunk the content, generate embeddings, and store the document data locally.
4. For a user question, classify intent, rewrite the query, retrieve relevant chunks, rerank results, and generate an answer with citations.

## Tech stack

- Python 3.11+
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Mistral AI API](https://docs.mistral.ai/)
- [NumPy](https://numpy.org/)
- [SQLite](https://docs.python.org/3/library/sqlite3.html)
- [Pytest](https://pytest.org/)

## Run locally

1. Install dependencies:

```bash
pip install -e ".[dev]"
```

2. Copy `.env.example` to `.env` and set `MISTRAL_API_KEY`.
3. Start the app:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the UI and `http://127.0.0.1:8000/docs` for the API docs.

## API

- `POST /ingest` - upload one or more PDF files
- `POST /query` - ask a question over indexed documents
- `GET /documents` - list ingested documents
- `DELETE /documents/{id}` - soft-delete a document
- `GET /health` - health check

## Storage

- `data/app.sqlite3` - document and chunk metadata
- `data/embeddings.npy` - embeddings matrix
- `data/bm25.json` - keyword index data

## Tests

The repo includes unit, integration, and frontend tests under `tests/`. The test suite also includes `tests/fakes/mistral.py`, a deterministic fake client used instead of live Mistral API calls.

Run the test suite with:

```bash
pytest -v
```

## Notes

- The project does not use an external search or RAG framework.
- The project does not use a third-party vector database.
