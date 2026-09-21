# 01 · RAG basics

## What RAG is
An LLM only knows its training data. **Retrieval-Augmented Generation** finds relevant text from *your* documents at question time and puts it in the prompt, so the answer is grounded in that text.

```
Indexing (once):   documents → chunk → embed → store in vector DB
Querying (each):   question → embed → find top-k similar chunks → prompt(question + chunks) → LLM → answer
```

## Key concepts

**Embedding** — a model turns text into a vector (list of numbers). Texts with similar *meaning* land close together.
`all-MiniLM-L6-v2` gives 384 numbers per text.

**Cosine similarity** — the angle between two vectors. 1 = same direction (similar meaning), 0 = unrelated.
Cosine *distance* = 1 − similarity, so **lower distance = more similar**.

**Chunking** — splitting documents into small pieces before embedding.
- Too big → one vector mixes many topics, retrieval gets fuzzy, and the prompt fills with noise.
- Too small → a chunk loses context ("it" with no idea what "it" is).
- **Overlap** repeats a few characters between neighbouring chunks so a sentence cut at a boundary isn't lost.
- Common start: 300–800 characters with 10–15 % overlap, then tune by testing.

**Vector store** — a database that stores vectors and quickly finds the nearest ones (Chroma, FAISS, pgvector, Qdrant). Uses approximate-nearest-neighbour indexes like HNSW so search stays fast with millions of chunks.

**Top-k** — how many chunks to retrieve. Low k can miss the answer; high k adds noise and cost.

**Grounded prompt** — tell the model to answer *only* from the context, cite chunk ids, and say "I don't know" otherwise. This reduces hallucination.

## Why RAG answers go wrong
| Failure | Symptom | First fix to try |
|---|---|---|
| Retrieval miss | Right answer exists but the chunk isn't in top-k | Better chunking, hybrid search (BM25 + vectors), reranker |
| Bad chunk boundaries | Answer split across two chunks | Increase overlap, chunk by headings |
| Lost in the middle | Right chunk retrieved but ignored | Fewer, better chunks; rerank so the best is first |
| Hallucination | Answer not supported by context | Stricter prompt, citations, faithfulness evaluation |
| Keyword mismatch | Codes/IDs/names not found | Hybrid search |

## Interview questions I should answer without notes
1. Why RAG instead of fine-tuning? *(fresh/private data, citations, cheaper, no retraining; fine-tuning changes style/behaviour, not knowledge updates)*
2. How do you choose chunk size?
3. What is the difference between a bi-encoder (embeddings) and a cross-encoder (reranker)?
4. How would you evaluate a RAG system?
5. How do you handle a document that updates every day?

## Notebook
- [naive_rag.ipynb](naive_rag.ipynb) — the full pipeline in ~30 lines, runs on Google Colab
