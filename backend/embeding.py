import os
import sys
import json
import argparse
import hashlib
import re
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

        # Clean answer-key artifacts
        passage = clean_medical_text(passage)

        subject_name = (x.get("subject_name") or "").strip()
        topic_name = (x.get("topic_name") or "").strip()
        
        # Normalize topic metadata - never leave empty
        if not topic_name:
            if subject_name:
                topic_name = subject_name  # Infer from subject
            else:
                topic_name = "Unknown"

        rows.append({
            "question": (x.get("question") or "").strip(),  # Stored in metadata only
            "exp": passage,  # Evidence/answer text only - becomes chunk content
            "subject_name": subject_name,
            "topic_name": topic_name,
            "id": str(x.get("id") or "").strip(),
        })
    return rows

def clean_medical_text(text: str) -> str:
    if not text:
        return text
    
    # Remove "Ans." prefixes (case insensitive, with optional punctuation)
    text = re.sub(r'\bans\.?\s*', '', text, flags=re.IGNORECASE)
    
    # Remove option letters like (a), (b), (c), etc. at start of lines or text
    text = re.sub(r'^\s*\([a-zA-Z]\)\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\b[a-zA-Z]\.\s*', '', text)  # Remove "a. ", "b. ", etc.
    
    # Remove page references like "P 640", "p. 123", "page 45"
    text = re.sub(r'\b[Pp]\.?\s*\d+', '', text)
    text = re.sub(r'\bpage\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bp\.\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpage\s+\d+', '', text, flags=re.IGNORECASE)
    
    # Remove edition references like "19th ed.", "2nd edition", etc.
    text = re.sub(r'\b\d+(?:st|nd|rd|th)?\s+(?:ed|edition)\.?\b', '', text, flags=re.IGNORECASE)
    
    # Remove reference markers like "Ref:", "Reference:", etc.
    text = re.sub(r'\b[Rr]ef\.?:', '', text)
    text = re.sub(r'\b[Rr]eference\.?:', '', text)
    
    # Remove extra whitespace and clean up
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove trailing dots and commas that might be left from removals
    text = re.sub(r'[.,]\s*$', '', text).strip()
    
    return text

# Chunking
def ends_with_sentence_boundary(text: str) -> bool:
    if not text:
        return False
    
    # Strip trailing whitespace and check last character
    text = text.rstrip()
    if not text:
        return False
    
    # Check for sentence-ending punctuation
    return text[-1] in '.!?'


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk and ends_with_sentence_boundary(chunk):
            chunks.append(chunk)

    return chunks


def create_chunked_rows(
    rows: List[Dict[str, Any]],
    chunk_size: int,
    overlap: int,
    min_words_for_merge: int = 20,
) -> List[Dict[str, Any]]:
    chunked = []
    total_chunks_created = 0
    total_chunks_dropped = 0
    total_chunks_merged = 0

    for row in rows:
        chunks = chunk_text(row["exp"], chunk_size, overlap)
        total_chunks_created += len(chunks)
        
        # Filter chunks that don't end with sentence boundaries
        valid_chunks = [chunk for chunk in chunks if ends_with_sentence_boundary(chunk)]
        total_chunks_dropped += len(chunks) - len(valid_chunks)
        
        # Merge small tail chunks with previous chunk
        if len(valid_chunks) > 1:
            # Check if last chunk is small and should be merged
            last_chunk = valid_chunks[-1]
            last_chunk_words = len(last_chunk.split())
            
            if last_chunk_words < min_words_for_merge:
                # Merge with previous chunk
                prev_chunk = valid_chunks[-2]
                merged_chunk = prev_chunk + " " + last_chunk
                
                # Replace the last two chunks with the merged one
                valid_chunks = valid_chunks[:-2] + [merged_chunk]
                total_chunks_merged += 1
        
        for idx, chunk in enumerate(valid_chunks):
            chunked.append({
                **row,
                "exp": chunk,
                "chunk_idx": idx,
            })

    print(f"    Chunking stats: {total_chunks_created} created, {total_chunks_dropped} dropped (incomplete), {total_chunks_merged} merged (small tails)")
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
            documents=[r["exp"] for r in batch],  # Evidence only - chunks read like textbook paragraphs
            embeddings=list(batch_embeddings),
            metadatas=[{
                "question": r["question"],  # Question stored in metadata only
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
    parser.add_argument("--min_words_merge", type=int, default=20, help="Minimum words for tail chunks before merging with previous")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    print(f"[1] Loading data: {args.data}")
    raw = load_json_or_jsonl(args.data)
    rows = normalize_rows(raw)

    if not rows:
        raise RuntimeError("No valid passages found to embed.")

    print(f"    Loaded {len(rows)} passages")

    print("[2] Chunking")
    chunked = create_chunked_rows(rows, args.chunk_size, args.overlap, args.min_words_merge)
    print(f"    Final result: {len(chunked)} valid chunks (ending with sentence boundaries)")

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