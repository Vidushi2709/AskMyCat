import os
import json
import argparse
import hashlib
from typing import List, Dict, Any

from tqdm import tqdm

import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


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


def build_id(row: Dict[str, Any]) -> str:
    """Prefer provided id; else deterministic hash of passage."""
    if row.get("id"):
        return str(row["id"])  # ensure string
    return hashlib.sha1((row.get("exp") or "").encode("utf-8")).hexdigest()


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
        metadatas = [{
            "question": r.get("question", ""),
            "exp": r.get("exp", ""),
            "subject_name": r.get("subject_name", ""),
            "topic_name": r.get("topic_name", ""),
            "id": r.get("id", ""),
            "source": "data/train",
        } for r in batch_rows]

        coll.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=list(batch_vecs),
        )


def main():
    ap = argparse.ArgumentParser(description="Embed data/train and store in a local Chroma collection")
    ap.add_argument("--data", default="data/train.json", help="Path to JSON/JSONL file")
    ap.add_argument("--chroma_path", default="./collections/ebm", help="Folder for persistent ChromaDB")
    ap.add_argument("--collection", default="ebm_passages", help="Collection name")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model")
    ap.add_argument("--batch", type=int, default=128, help="Embedding batch size")
    ap.add_argument("--device", default="cuda", help="Device to use: cuda or cpu (default: cuda)")
    args = ap.parse_args()

    print(f"Loading dataset from {args.data} ...")
    rows = load_dataset(args.data)
    if not rows:
        print("No rows to embed (need exp). Abort.")
        return
    print(f"Loaded {len(rows)} rows")

    print(f"Embedding with {args.model} on {args.device} ...")
    vectors = embed_passages(rows, args.model, args.batch, args.device)

    print(f"Connecting to Chroma at {args.chroma_path} ...")
    client = connect_chroma(args.chroma_path)

    print(f"Ingesting into collection '{args.collection}' ...")
    ingest_chroma(client, rows, vectors, args.collection, batch_size=512)

    print("✓ Done.")


if __name__ == "__main__":
    main()
