"""Storage layer for the StackAI RAG application.

This package manages persistent storage for document metadata (SQLite),
vector embeddings (NumPy), and keyword indices (BM25). It implements
an atomic 'stage-then-publish' pattern to ensure data integrity during
concurrent updates and recovery after crashes.
"""
