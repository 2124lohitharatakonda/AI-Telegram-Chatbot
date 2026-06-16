"""
FAISS Vector Store — Document Indexing & Similarity Search
Embeds documents using sentence-level TF-IDF and indexes with FAISS IndexFlatL2.
"""

import os
import json
import numpy as np
import faiss
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

INDEX_PATH = "models/faiss.index"
CHUNKS_PATH = "models/doc_chunks.json"
EMBED_PATH = "models/embedder.pkl"

EMBEDDING_DIM = 256


# ---------------------------------------------------------------------------
# Document Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def load_documents(folder: str = "documents") -> list[dict]:
    docs = []
    if not os.path.exists(folder):
        # demo documents
        return [
            {"source": "faq.txt",           "text": "Our refund policy allows returns within 30 days. Items must be unused and in original packaging. Refunds are processed within 5–7 business days. Damaged items must be reported within 48 hours with photo evidence."},
            {"source": "x200_manual.txt",    "text": "Model X200 User Guide v3.2. Installation: Connect power adapter. Press power button for 3 seconds. Default WiFi password is printed on the device label. Warranty: 2 years from date of purchase."},
            {"source": "shipping_policy.txt","text": "Standard shipping takes 3–5 business days. Express shipping is available for additional cost. Free shipping on orders above ₹500. International shipping available to 45 countries."},
            {"source": "support.txt",        "text": "Support hours: Monday to Friday, 9 AM to 6 PM IST. Contact: support@nexbot.in or call 1800-XXX-XXXX. Average response time is under 2 hours during business hours."},
        ]

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                docs.append({"source": filename, "text": f.read()})
    return docs


# ---------------------------------------------------------------------------
# Embedding Pipeline
# ---------------------------------------------------------------------------

def build_embedder() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, sublinear_tf=True, ngram_range=(1, 2))),
        ("svd",   TruncatedSVD(n_components=EMBEDDING_DIM, random_state=42)),
    ])


def embed_texts(texts: list[str], embedder: Pipeline, fit: bool = False) -> np.ndarray:
    if fit:
        vecs = embedder.fit_transform(texts)
    else:
        vecs = embedder.transform(texts)
    # L2-normalise for cosine similarity via IndexFlatL2
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return (vecs / norms).astype("float32")


# ---------------------------------------------------------------------------
# Index Build
# ---------------------------------------------------------------------------

def build_index(docs: list[dict] = None) -> tuple:
    if docs is None:
        docs = load_documents()

    all_chunks, metadata = [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({"source": doc["source"], "chunk_id": i, "text": chunk})

    embedder = build_embedder()
    vectors = embed_texts(all_chunks, embedder, fit=True)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(vectors)

    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    joblib.dump(embedder, EMBED_PATH)

    print(f"FAISS index built: {index.ntotal} vectors · dim={EMBEDDING_DIM}")
    return index, metadata, embedder


def load_index():
    index    = faiss.read_index(INDEX_PATH)
    embedder = joblib.load(EMBED_PATH)
    with open(CHUNKS_PATH) as f:
        metadata = json.load(f)
    return index, metadata, embedder


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(query: str, index, metadata: list, embedder: Pipeline, top_k: int = 3) -> list[dict]:
    q_vec = embed_texts([query], embedder, fit=False)
    distances, indices = index.search(q_vec, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue
        score = float(1 / (1 + dist))
        results.append({
            "rank":       rank + 1,
            "score":      round(score, 4),
            "source":     metadata[idx]["source"],
            "chunk_id":   metadata[idx]["chunk_id"],
            "text":       metadata[idx]["text"][:300],
        })
    return results


def format_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        parts.append(f"[Source: {r['source']} | Score: {r['score']}]\n{r['text']}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    print("Building FAISS index from demo documents...")
    index, metadata, embedder = build_index()

    queries = [
        "what is the refund policy for damaged items",
        "how to set up Model X200",
        "how do I contact support",
    ]

    print("\n=== FAISS Retrieval Demo ===")
    for q in queries:
        results = search(q, index, metadata, embedder, top_k=2)
        print(f"\nQuery: {q}")
        for r in results:
            print(f"  [{r['rank']}] {r['source']} (score={r['score']}) — {r['text'][:80]}...")
