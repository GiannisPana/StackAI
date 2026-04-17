## 3. Technical part

Develop a simple Python backend for a Retrieval-Augmented Generation (RAG) pipeline, using a Language Model (LLM) for a knowledge base consisting of PDF files.

Key Components:

- Data Ingestion:
    - Develop an API endpoint to upload a set of files.
    - Implement a text extraction and chunking algorithm from PDF files. Write down your considerations.
- Query Processing, given a user query:
    - Detect the intent of the query to assess if triggering a knowledge base search is necessary. For example, “hello” should not trigger a search.
    - Transform the query to improve the retrieving for RAG.
- Semantic Search:
    - Design a search mechanism over the ingested files using the processed query.
    - How would you combine both semantic and keyword results?
- Post-processing:
    - Merge and re-rank the results to improve retrieval performance.
- Generation:
    - Call a language model with a prompt template to generate answers to the user questions over the ingested knowledge base.
- UI to chat with your system


Deliverable:

- A Github repo with:
    - A README.md detailing the system design, with a balance of text, and diagrams if necessary, explaining how the system operates and how to run it.
    - Python implementation of the pipeline using FastAPI with the following set of endpoints for:
        - Ingestion:
            - One endpoint to send one or more PDF files for ingestion
        - Querying:
            - One endpoint to query the system with user questions.
    - Links to specific libraries and software you used.
    - UI
    - Commit history.

Requirements and limitations:

- FastAPI
- Mistral AI API (https://docs.mistral.ai/)
    - *Note:* you can use the API key xxx
    - If you see it doesn’t work, then use your own

Extra considerations:

- You do NOT make use of **any** external library for **search**, or **RAG**

Bonus points:

- You do not use any third-party vector database
- Additional features:
    - **Citations required**: refuse to answer if top-k chunks don’t meet a similarity threshold; return “insufficient evidence”.
    - **Answer shaping:** switch templates by intent; structured outputs for lists/tables.
    - **Hallucination filters:** post-hoc “evidence check” that scans answer sentences for non-supported claims.
    - **Query refusal policies**: PII, legal/medical disclaimers.


Evaluation criteria:

- Quality of retrieval and results
- Organization and readability of code
- Problem thinking
- Security
- Scalability