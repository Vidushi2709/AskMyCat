import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Allow importing config from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TRAIN_DATA,
    CHROMA_PATH,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    OVERLAP,
    EMBEDDING_BATCH_SIZE,
)

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSON array or JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return rows


def normalize_rows(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize expected schema and filter empty passages."""
    rows = []
    for x in data:
        passage = (x.get("exp") or "").strip()
        if not passage:
            continue

        rows.append({
            "question": (x.get("question") or "").strip(),
            "exp": passage,
            "subject_name": (x.get("subject_name") or "").strip(),
            "topic_name": (x.get("topic_name") or "").strip(),
            "id": str(x.get("id") or "").strip(),
        })
    return rows

# Chunking
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Character-based sliding window chunking."""
    if not text:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def create_chunked_rows(
    rows: List[Dict[str, Any]],
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    chunked = []

    for row in rows:
        chunks = chunk_text(row["exp"], chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            chunked.append({
                **row,
                "exp": chunk,
                "chunk_idx": idx,
            })

    return chunked

def embed_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str,
):
    model = SentenceTransformer(model_name, device=device)
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

# Chroma
def connect_chroma(path: str) -> PersistentClient:
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def get_or_create_collection(client: PersistentClient, name: str):
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name)


def make_chunk_id(row: Dict[str, Any]) -> str:
    base = row["id"]
    if not base:
        base = hashlib.sha1(row["exp"].encode("utf-8")).hexdigest()
    return f"{base}_chunk_{row['chunk_idx']}"


def ingest(
    client: PersistentClient,
    collection_name: str,
    rows: List[Dict[str, Any]],
    embeddings,
    batch_size: int,
):
    collection = get_or_create_collection(client, collection_name)

    for i in tqdm(range(0, len(rows), batch_size), desc="Upserting to Chroma"):
        batch = rows[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        collection.add(
            ids=[make_chunk_id(r) for r in batch],
            documents=[r["exp"] for r in batch],
            embeddings=list(batch_embeddings),
            metadatas=[{
                "question": r["question"],
                "subject_name": r["subject_name"],
                "topic_name": r["topic_name"],
                "original_id": r["id"],
                "chunk_idx": r["chunk_idx"],
                "source": "train_data",
            } for r in batch],
        )

# Main
def main():
    parser = argparse.ArgumentParser(description="Embed text and ingest into ChromaDB")

    parser.add_argument("--data", default=str(TRAIN_DATA))
    parser.add_argument("--chroma_path", default=str(CHROMA_PATH))
    parser.add_argument("--collection", default=CHROMA_COLLECTION_NAME)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch", type=int, default=EMBEDDING_BATCH_SIZE)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    print(f"[1] Loading data: {args.data}")
    raw = load_json_or_jsonl(args.data)
    rows = normalize_rows(raw)

    if not rows:
        raise RuntimeError("No valid passages found to embed.")

    print(f"    Loaded {len(rows)} passages")

    print("[2] Chunking")
    chunked = create_chunked_rows(rows, args.chunk_size, args.overlap)
    print(f"    Created {len(chunked)} chunks")

    print("[3] Embedding")
    embeddings = embed_texts(
        texts=[r["exp"] for r in chunked],
        model_name=args.model,
        batch_size=args.batch,
        device=args.device,
    )

    print("[4] Connecting to Chroma")
    client = connect_chroma(args.chroma_path)

    print("[5] Ingesting")
    ingest(
        client=client,
        collection_name=args.collection,
        rows=chunked,
        embeddings=embeddings,
        batch_size=512,
    )

    print("✓ Done")


if __name__ == "__main__":
    main()