import os
import json
import argparse
import hashlib
from typing import List, Dict, Any
import sys
from pathlib import Path

from tqdm import tqdm

import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_DATA, CHROMA_PATH, CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL, CHUNK_SIZE, OVERLAP, EMBEDDING_BATCH_SIZE
)


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load JSON (array) or JSONL and normalize fields.
    Expects keys: question, exp, subject_name, topic_name, id.
    Returns list of rows with these keys (missing values become empty strings).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # If wrapped in a dict, extract the first list value
    if isinstance(data, dict):
        for _, v in data.items():
            if isinstance(v, list):
                data = v
                break

    rows: List[Dict[str, Any]] = []
    for x in data or []:
        q = (x.get("question") or "").strip()
        p = (x.get("exp") or "").strip()
        subj = (x.get("subject_name") or "").strip()
        topic = (x.get("topic_name") or "").strip()
        item_id = (x.get("id") or "").strip()
        if p:  # require passage to embed
            rows.append({
                "question": q,
                "exp": p,
                "subject_name": subj,
                "topic_name": topic,
                "id": item_id,
            })
    return rows

def chunk_passage(text: str, chunk_size: int = 512, overlap_size: int = 128):
    """Split text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunk_stripped = chunk.strip()
        if chunk_stripped:  # Only add non-empty chunks
            chunks.append(chunk_stripped)
        # Move forward by (chunk_size - overlap) for sliding window
        start += chunk_size - overlap_size
    return chunks

def build_id(row: Dict[str, Any]) -> str:
    base_id = row.get("id") or hashlib.sha1((row.get("exp") or "").encode("utf-8")).hexdigest()
    chunk_idx = row.get("chunk_idx", 0)
    return f"{base_id}_chunk_{chunk_idx}"


def embed_passages(rows: List[Dict[str, Any]], model_name: str, batch_size: int, device: str):
    model = SentenceTransformer(model_name, device=device)
    texts = [r["exp"] for r in rows]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors


def connect_chroma(persist_path: str) -> PersistentClient:
    os.makedirs(persist_path, exist_ok=True)
    return chromadb.PersistentClient(path=persist_path)


def get_collection(client: PersistentClient, name: str):
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name)


def ingest_chroma(client: PersistentClient, rows: List[Dict[str, Any]], vectors, collection_name: str, batch_size: int = 512):
    coll = get_collection(client, collection_name)

    total = len(rows)
    for i in tqdm(range(0, total, batch_size), desc="Chroma upsert"):
        batch_rows = rows[i:i + batch_size]
        batch_vecs = vectors[i:i + batch_size]

        ids = [build_id(r) for r in batch_rows]
        documents = [r["exp"] for r in batch_rows]
        # Metadata: DON'T include "exp" (passage), just reference fields
        metadatas = [{
            "question": r.get("question", ""),
            "subject_name": r.get("subject_name", ""),
            "topic_name": r.get("topic_name", ""),
            "id": r.get("id", ""),
            "chunk_idx": r.get("chunk_idx", 0),
            "source": "data/train",
        } for r in batch_rows]

        coll.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=list(batch_vecs),
        )

def prepare_chunked_data(rows: List[Dict[str, Any]], chunk_size: int = 512, overlap: int = 128) -> List[Dict[str, Any]]:
    """Create overlapping chunks from passages, preserving metadata."""
    chunked_rows = []
    for row in rows: 
        chunks = chunk_passage(row['exp'], chunk_size=chunk_size, overlap_size=overlap)
        for chunk_idx, chunk_text in enumerate(chunks):
            chunked_rows.append({
                "question": row.get("question", ""),
                "exp": chunk_text,
                "subject_name": row.get("subject_name", ""),
                "topic_name": row.get("topic_name", ""),
                "id": row.get("id", ""),
                "chunk_idx": chunk_idx,
            })
    return chunked_rows

def main():
    ap = argparse.ArgumentParser(description="Embed chunked data/train and store in a local Chroma collection")
    ap.add_argument("--data", default=str(TRAIN_DATA), help="Path to JSON/JSONL file")
    ap.add_argument("--chroma_path", default=str(CHROMA_PATH), help="Folder for persistent ChromaDB")
    ap.add_argument("--collection", default=CHROMA_COLLECTION_NAME, help="Collection name")
    ap.add_argument("--model", default=EMBEDDING_MODEL, help="SentenceTransformer model")
    ap.add_argument("--batch", type=int, default=EMBEDDING_BATCH_SIZE, help="Embedding batch size")
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Chunk size in characters")
    ap.add_argument("--overlap", type=int, default=OVERLAP, help="Overlap between chunks")
    ap.add_argument("--device", default="cuda", help="Device to use: cuda or cpu (default: cuda)")
    args = ap.parse_args()

    print(f"Loading dataset from {args.data} ...")
    rows = load_dataset(args.data)
    if not rows:
        print("No rows to embed (need exp). Abort.")
        return
    print(f"Loaded {len(rows)} rows")

    print(f"Chunking passages (chunk_size={args.chunk_size}, overlap={args.overlap}) ...")
    chunked_rows = prepare_chunked_data(rows, args.chunk_size, args.overlap)
    print(f"Created {len(chunked_rows)} chunks from {len(rows)} passages")

    print(f"Embedding with {args.model} on {args.device} ...")
    vectors = embed_passages(chunked_rows, args.model, args.batch, args.device)
    
    print(f"Connecting to Chroma at {args.chroma_path} ...")
    client = connect_chroma(args.chroma_path)

    print(f"Ingesting into collection '{args.collection}' ...")
    ingest_chroma(client, chunked_rows, vectors, args.collection, batch_size=512)

    print(" Done.")


if __name__ == "__main__":
    main()