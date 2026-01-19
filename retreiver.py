from chromadb import PersistentClient
import torch
from transformers import AutoTokenizer, AutoModel
from train import TransformerScorer  

# Use chroma_db as the persistent path
client = PersistentClient(path="./chroma_db")
try:
    coll = client.get_collection("passages")
except Exception as e:
    print("Collection 'ebm_passages' not found in ./chroma_db.")
    print("Available collections:", [c.name for c in client.list_collections()])
    raise e

query = "What are the causes of renal pelvis dilatation?"
results = coll.query(query_texts=[query], n_results=20)
candidates = results["documents"][0]  # list of passages
metadatas = results["metadatas"][0]   # list of dicts with question, exp, subject_name, topic_name, id

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name)

# Load trained weights
model = TransformerScorer(encoder).to("cuda")
model.load_state_dict(torch.load("scorer.ckpt", map_location="cuda"))
model.eval()

def score_passages(model, tokenizer, query, passages, device="cuda", batch_size=32):
    scores = []
    with torch.no_grad(), torch.amp.autocast(device):
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i+batch_size]
            enc = tokenizer([query]*len(batch), batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(enc)
            energy = torch.sigmoid(logits)  # lower = better
            scores.extend(energy.cpu().tolist())
    return scores

# Score and rerank according to the energy function
scores = score_passages(model, tokenizer, query, candidates)
reranked = sorted(zip(candidates, metadatas, scores), key=lambda x: x[2])

print("Top reranked passages:")
for i, (passage, meta, score) in enumerate(reranked[:5], 1):
    print(f"{i}. energy={score:.3f} | {passage[:200]}{'...' if len(passage)>200 else ''}")
    print(f"   meta: {meta}\n")