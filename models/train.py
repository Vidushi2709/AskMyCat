"""
Training script for EBM Ranking Scorer model.
Converts ranking model from notebook to production training script with evaluation metrics.
"""
import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)


class RankingScorer(nn.Module):
    """Ranking model: projects embeddings to ranking space."""
    def __init__(self, encoder, embedding_dim=128):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = encoder.config.hidden_size
        
        # Project to embedding space for ranking
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, enc_ips):
        """Return embedding vector for ranking."""
        out = self.encoder(**enc_ips).last_hidden_state
        
        # Mean pooling over tokens
        mask = enc_ips["attention_mask"].unsqueeze(-1)  # [B, L, 1]
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, H]
        
        # Project to embedding space
        embedding = self.projection(pooled)  # [B, embedding_dim]
        
        return embedding


def extract_medical_topics(text):
    """Extract medical entities - simple keyword matching.
    TODO: Upgrade to scispacy or BioBERT NER for production.
    """
    medical_keywords = {
        'diabetes', 'hypertension', 'aspirin', 'metformin', 
        'surgery', 'pregnancy', 'cancer', 'antibiotics',
        'heart', 'blood pressure', 'cholesterol', 'infection',
        'pain', 'fever', 'asthma', 'copd', 'stroke', 'mi',
        'myocardial infarction', 'coronary', 'cardiovascular'
    }
    
    text_lower = text.lower()
    return [kw for kw in medical_keywords if kw in text_lower]


def has_contradiction(text1, text2):
    """Check if two passages contradict each other.
    Simple heuristic - improve with NLI model for production.
    """
    contradiction_pairs = [
        ('safe', 'contraindicated'),
        ('effective', 'ineffective'),
        ('recommended', 'not recommended'),
        ('increases', 'decreases'),
        ('approved', 'not approved'),
        ('use', 'avoid'),
        ('beneficial', 'harmful'),
        ('normal', 'abnormal')
    ]
    
    text1_lower = text1.lower()
    text2_lower = text2.lower()
    
    for pos, neg in contradiction_pairs:
        if (pos in text1_lower and neg in text2_lower) or \
           (neg in text1_lower and pos in text2_lower):
            return True
    return False


class DataProcessor:
    """Load and process training data."""
    
    @staticmethod
    def load_dataset(data_path):
        """Load JSON or JSONL data."""
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            raw = f.read()
        
        try:
            dataset = json.loads(raw)
        except json.JSONDecodeError:
            dataset = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        # Handle if dataset is a dict with a list inside
        if isinstance(dataset, dict):
            for key, value in dataset.items():
                if isinstance(value, list):
                    dataset = value
                    break
        
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset
    
    @staticmethod
    def build_easy_negatives(dataset, max_attempts=10):
        """Sample random negative passages (easy negatives)."""
        logger.info("Building easy negatives...")
        all_passages = [x.get("exp") for x in dataset if x.get("exp")]
        easy_negatives = []
        
        for item in dataset:
            q = item.get("question")
            gold = item.get("exp")
            if not q or not gold or not all_passages:
                continue
            
            candidate = gold
            tries = 0
            while candidate == gold and tries < max_attempts:
                candidate = random.choice(all_passages)
                tries += 1
            
            if candidate != gold:
                easy_negatives.append((q, candidate))
        
        logger.info(f"Built {len(easy_negatives)} easy negatives")
        return easy_negatives
    
    @staticmethod
    def build_hard_negatives(dataset, sample_size=1000):
        """Use BM25 to find hard negatives."""
        logger.info("Building hard negatives using BM25...")
        all_passages = [x.get("exp") for x in dataset if x.get("exp")]
        
        # Sample to speed up BM25
        if len(all_passages) > sample_size:
            sampled_idx = random.sample(range(len(all_passages)), sample_size)
            all_passages = [all_passages[i] for i in sampled_idx]
        
        # Build BM25 index
        corpus_tokens = [p.lower().split() for p in all_passages]
        bm25 = BM25Okapi(corpus_tokens)
        
        hard_negatives = []
        for item in dataset:
            q = item.get("question")
            gold = item.get("exp")
            if not q or not gold:
                continue
            
            q_tokens = q.lower().split()
            scores = bm25.get_scores(q_tokens)
            ranked = sorted(zip(all_passages, scores), key=lambda x: x[1], reverse=True)
            
            # Take top non-gold passage
            for passage, _ in ranked:
                if passage != gold:
                    hard_negatives.append((q, passage))
                    break
        
        logger.info(f"Built {len(hard_negatives)} hard negatives")
        return hard_negatives
    
    @staticmethod
    def build_medical_hard_negatives(dataset, sample_size=1000):
        """Build negatives that are medically similar but incorrect.
        These are the most valuable negatives for medical domain!
        """
        logger.info("Building medical hard negatives...")
        
        # Group passages by medical topic/specialty
        topic_groups = defaultdict(list)
        for item in dataset:
            question = item.get("question", "")
            passage = item.get("exp", "")
            
            if not passage:
                continue
            
            # Extract medical topics from question
            topics = extract_medical_topics(question)
            
            for topic in topics:
                topic_groups[topic].append(passage)
        
        logger.info(f"Grouped passages into {len(topic_groups)} medical topics")
        
        medical_hard_negatives = []
        for item in dataset:
            q = item.get("question")
            gold = item.get("exp")
            if not q or not gold:
                continue
            
            topics = extract_medical_topics(q)
            
            # Sample from SAME medical topic but different passage
            for topic in topics:
                candidates = [p for p in topic_groups.get(topic, []) if p != gold]
                if candidates:
                    hard_neg = random.choice(candidates)
                    medical_hard_negatives.append((q, hard_neg))
                    break
        
        logger.info(f"Built {len(medical_hard_negatives)} medical hard negatives")
        return medical_hard_negatives
    
    @staticmethod
    def build_contradiction_negatives(dataset):
        """Build negatives that contradict the correct answer.
        Critical for preventing hallucination!
        """
        logger.info("Building contradiction negatives...")
        
        contradiction_negatives = []
        
        for item in dataset:
            q = item.get("question")
            gold = item.get("exp")
            
            if not q or not gold:
                continue
            
            # Look for passages that discuss same topic but contradict
            for other_item in dataset:
                other_passage = other_item.get("exp")
                
                if other_passage == gold or not other_passage:
                    continue
                
                # Check if they have same topics but contradict
                gold_topics = extract_medical_topics(gold)
                other_topics = extract_medical_topics(other_passage)
                
                # Must share at least one topic
                if gold_topics and other_topics and \
                   any(t in other_topics for t in gold_topics):
                    if has_contradiction(gold, other_passage):
                        contradiction_negatives.append((q, other_passage))
                        break
        
        logger.info(f"Built {len(contradiction_negatives)} contradiction negatives")
        return contradiction_negatives
    
    @staticmethod
    def create_triplets(positives, negatives):
        """Create triplet samples (query, positive, negative)."""
        logger.info("Creating triplet samples...")
        positives_clean = [(q, p) for (q, p) in positives if q and p]
        negatives_clean = [(q, p) for (q, p) in negatives if q and p]
        
        # Build negative index
        neg_by_query = defaultdict(list)
        for q, p in negatives_clean:
            neg_by_query[q].append(p)
        
        logger.info(f"Built negative index for {len(neg_by_query)} unique queries")
        
        # Create triplets
        triplet_samples = []
        for q, pos_p in tqdm(positives_clean, desc="Creating triplets"):
            neg_for_q = neg_by_query.get(q, [])
            neg_for_q = [p for p in neg_for_q if p != pos_p]
            
            if neg_for_q:
                neg_p = random.choice(neg_for_q)
                triplet_samples.append((q, pos_p, neg_p))
        
        random.shuffle(triplet_samples)
        logger.info(f"Created {len(triplet_samples)} triplet samples")
        return triplet_samples


class DataAugmentation:
    """Data augmentation for medical queries and passages."""
    
    @staticmethod
    def augment_medical_query(query, num_augments=2):
        """Augment queries with medical synonyms and paraphrasing."""
        augmented = [query]
        
        # Medical synonym replacement
        medical_synonyms = {
            'heart attack': ['myocardial infarction', 'MI', 'cardiac event'],
            'high blood pressure': ['hypertension', 'elevated BP'],
            'diabetes': ['diabetes mellitus', 'DM'],
            'medicine': ['medication', 'drug', 'pharmaceutical'],
            'doctor': ['physician', 'clinician'],
            'symptoms': ['signs', 'manifestations'],
            'treatment': ['therapy', 'management'],
            'side effects': ['adverse effects', 'complications'],
        }
        
        query_lower = query.lower()
        for original, synonyms in medical_synonyms.items():
            if original in query_lower:
                for syn in synonyms[:num_augments]:
                    new_query = query_lower.replace(original, syn)
                    if new_query != query_lower:
                        augmented.append(new_query.capitalize())
        
        # Question reformulation
        reformulations = {
            'What is': ['What are', 'Can you explain', 'Define'],
            'How does': ['How do', 'What is the mechanism of'],
            'What are': ['What is', 'List the'],
        }
        
        for original, alternatives in reformulations.items():
            if original in query:
                for alt in alternatives[:num_augments]:
                    new_query = query.replace(original, alt, 1)
                    if new_query != query:
                        augmented.append(new_query)
        
        return list(set(augmented))[:num_augments + 1]
    
    @staticmethod
    def augment_passage(passage, drop_rate=0.1):
        """Augment passages by randomly dropping words (noise injection)."""
        words = passage.split()
        num_keep = int(len(words) * (1 - drop_rate))
        
        if num_keep < len(words) and num_keep > 0:
            indices = random.sample(range(len(words)), num_keep)
            indices.sort()
            return ' '.join([words[i] for i in indices])
        
        return passage


def create_triplets_with_augmentation(positives, negatives):
    """Create triplets with data augmentation for 3x more training data."""
    logger.info("Creating triplets with data augmentation...")
    
    positives_clean = [(q, p) for (q, p) in positives if q and p]
    negatives_clean = [(q, p) for (q, p) in negatives if q and p]
    
    # Build negative index
    neg_by_query = defaultdict(list)
    for q, p in negatives_clean:
        neg_by_query[q].append(p)
    
    triplet_samples = []
    for q, pos_p in tqdm(positives_clean, desc="Creating augmented triplets"):
        # Get negatives for this query
        neg_for_q = neg_by_query.get(q, [])
        neg_for_q = [p for p in neg_for_q if p != pos_p]
        
        if not neg_for_q:
            continue
        
        # Original triplet
        neg_p = random.choice(neg_for_q)
        triplet_samples.append((q, pos_p, neg_p))
        
        # Augmented versions
        aug_queries = DataAugmentation.augment_medical_query(q, num_augments=2)
        for aug_q in aug_queries[1:]:  # Skip first (original)
            aug_pos = DataAugmentation.augment_passage(pos_p)
            neg_p = random.choice(neg_for_q)
            triplet_samples.append((aug_q, aug_pos, neg_p))
    
    random.shuffle(triplet_samples)
    logger.info(f"Created {len(triplet_samples)} augmented triplet samples")
    return triplet_samples


def collate_batch_triplet(batch, tokenizer, max_len=128):
    """Collate triplet batch."""
    qs, pos_ps, neg_ps = zip(*batch)
    
    # Encode queries
    enc_q = tokenizer(
        list(qs),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    # Encode positive passages
    enc_pos = tokenizer(
        list(pos_ps),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    # Encode negative passages
    enc_neg = tokenizer(
        list(neg_ps),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    return enc_q, enc_pos, enc_neg


class Trainer:
    """Training and evaluation logic with advanced features."""
    
    def __init__(self, model, tokenizer, device, output_dir="checkpoints"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = {
            "train_loss": [], 
            "val_acc": [], 
            "val_mrr": [],
            "val_recall@5": [],
            "val_precision@1": []
        }
        self.negative_cache = []  # For online hard negative mining
    
    def train_epoch(self, train_loader, optimizer, scaler, epoch, margin=0.5):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for enc_q, enc_pos, enc_neg in pbar:
            enc_q = {k: v.to(self.device) for k, v in enc_q.items()}
            enc_pos = {k: v.to(self.device) for k, v in enc_pos.items()}
            enc_neg = {k: v.to(self.device) for k, v in enc_neg.items()}
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                emb_q = self.model(enc_q)       # Query embeddings
                emb_pos = self.model(enc_pos)   # Positive passage embeddings
                emb_neg = self.model(enc_neg)   # Negative passage embeddings
                
                # Cosine similarity: query vs positive, query vs negative
                pos_sim = F.cosine_similarity(emb_q, emb_pos)  # [B]
                neg_sim = F.cosine_similarity(emb_q, emb_neg)  # [B]
                
                # Triplet loss: max(0, margin - (pos_sim - neg_sim))
                loss = torch.clamp(margin - (pos_sim - neg_sim), min=0).mean()
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        self.metrics["train_loss"].append(avg_loss)
        logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
        return avg_loss
    
    def train_epoch_mnr(self, train_loader, optimizer, scaler, epoch, temperature=0.05):
        """Train with Multiple Negatives Ranking Loss (better than triplet).
        
        MNR uses in-batch negatives, making training more efficient and effective.
        """
        self.model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [MNR]")
        
        for enc_q, enc_pos, enc_neg in pbar:
            enc_q = {k: v.to(self.device) for k, v in enc_q.items()}
            enc_pos = {k: v.to(self.device) for k, v in enc_pos.items()}
            enc_neg = {k: v.to(self.device) for k, v in enc_neg.items()}
            
            batch_size = enc_q['input_ids'].shape[0]
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                emb_q = self.model(enc_q)       # [B, D]
                emb_pos = self.model(enc_pos)   # [B, D]
                emb_neg = self.model(enc_neg)   # [B, D]
                
                # In-batch negatives: use all positives as negatives for other queries
                # This gives us B-1 negatives per query for free!
                scores = torch.matmul(emb_q, emb_pos.T) / temperature  # [B, B]
                
                # Labels: diagonal elements are the positives
                labels = torch.arange(batch_size).to(self.device)
                
                # Cross-entropy: model should rank correct positive highest
                loss = F.cross_entropy(scores, labels)
                
                # Add explicit hard negatives for extra supervision
                neg_scores = F.cosine_similarity(emb_q, emb_neg) / temperature
                pos_scores = scores.diagonal()
                
                # Margin loss between positive and explicit negative
                margin_loss = F.relu(0.2 - (pos_scores - neg_scores)).mean()
                loss = loss + 0.5 * margin_loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        self.metrics["train_loss"].append(avg_loss)
        logger.info(f"Epoch {epoch} - Avg MNR Loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(self, val_loader):
        """Evaluate model on validation set with multiple metrics."""
        self.model.eval()
        correct = total = 0
        mrr_scores = []
        
        with torch.no_grad():
            for enc_q, enc_pos, enc_neg in tqdm(val_loader, desc="Validation"):
                enc_q = {k: v.to(self.device) for k, v in enc_q.items()}
                enc_pos = {k: v.to(self.device) for k, v in enc_pos.items()}
                enc_neg = {k: v.to(self.device) for k, v in enc_neg.items()}
                
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    emb_q = self.model(enc_q)
                    emb_pos = self.model(enc_pos)
                    emb_neg = self.model(enc_neg)
                
                # Ranking accuracy: positive similarity > negative similarity
                pos_sim = F.cosine_similarity(emb_q, emb_pos)
                neg_sim = F.cosine_similarity(emb_q, emb_neg)
                correct += (pos_sim > neg_sim).sum().item()
                total += emb_q.shape[0]
                
                # MRR: Mean Reciprocal Rank
                # For each sample, rank positive against negative
                for p_sim, n_sim in zip(pos_sim, neg_sim):
                    # Rank positive: 1 if pos_sim > neg_sim, 2 otherwise
                    rank = 1 if p_sim > n_sim else 2
                    mrr_scores.append(1.0 / rank)
        
        accuracy = correct / total if total > 0 else 0
        mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
        
        self.metrics["val_acc"].append(accuracy)
        self.metrics["val_mrr"].append(mrr)
        
        logger.info(f"Validation - Accuracy: {accuracy:.3f}, MRR: {mrr:.3f}")
        return accuracy, mrr
    
    def evaluate_comprehensive(self, val_loader):
        """Comprehensive evaluation with multiple metrics.
        
        Returns:
            dict: accuracy, MRR, recall@5, precision@1
        """
        self.model.eval()
        
        metrics = {
            'correct': 0,
            'total': 0,
            'mrr': [],
            'recall@5': [],
        }
        
        with torch.no_grad():
            for enc_q, enc_pos, enc_neg in tqdm(val_loader, desc="Comprehensive Eval"):
                enc_q = {k: v.to(self.device) for k, v in enc_q.items()}
                enc_pos = {k: v.to(self.device) for k, v in enc_pos.items()}
                enc_neg = {k: v.to(self.device) for k, v in enc_neg.items()}
                
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    emb_q = self.model(enc_q)
                    emb_pos = self.model(enc_pos)
                    emb_neg = self.model(enc_neg)
                
                batch_size = emb_q.shape[0]
                
                # Evaluate each sample in batch
                for i in range(batch_size):
                    q_emb = emb_q[i:i+1]
                    
                    # Combine positive and negative
                    candidates = torch.cat([emb_pos[i:i+1], emb_neg[i:i+1]], dim=0)
                    
                    # Compute similarities
                    sims = F.cosine_similarity(q_emb, candidates, dim=1)
                    
                    # Rank: 1 if positive > negative, 2 otherwise
                    rank = 2 - (sims[0] > sims[1]).long().item()
                    
                    # Accuracy (precision@1)
                    if rank == 1:
                        metrics['correct'] += 1
                    
                    # MRR
                    metrics['mrr'].append(1.0 / rank)
                    
                    # Recall@5 (always 1.0 in this binary case if rank <= 5)
                    metrics['recall@5'].append(1.0 if rank <= 5 else 0.0)
                    
                    metrics['total'] += 1
        
        # Aggregate
        results = {
            'accuracy': metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0,
            'mrr': sum(metrics['mrr']) / len(metrics['mrr']) if metrics['mrr'] else 0,
            'recall@5': sum(metrics['recall@5']) / len(metrics['recall@5']) if metrics['recall@5'] else 0,
            'precision@1': metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
        }
        
        # Update metrics tracking
        self.metrics['val_acc'].append(results['accuracy'])
        self.metrics['val_mrr'].append(results['mrr'])
        self.metrics['val_recall@5'].append(results['recall@5'])
        self.metrics['val_precision@1'].append(results['precision@1'])
        
        logger.info("Comprehensive Evaluation Results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def save_model(self, checkpoint_name="ranking_scorer.ckpt"):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / checkpoint_name
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")
    
    def print_metrics_summary(self):
        """Print training metrics summary."""
        logger.info("\n" + "="*60)
        logger.info("TRAINING METRICS SUMMARY")
        logger.info("="*60)
        
        if self.metrics["train_loss"]:
            logger.info(f"Final Train Loss: {self.metrics['train_loss'][-1]:.4f}")
            logger.info(f"Best Train Loss: {min(self.metrics['train_loss']):.4f}")
        
        if self.metrics["val_acc"]:
            logger.info(f"Final Val Accuracy: {self.metrics['val_acc'][-1]:.3f}")
            logger.info(f"Best Val Accuracy: {max(self.metrics['val_acc']):.3f}")
        
        if self.metrics["val_mrr"]:
            logger.info(f"Final Val MRR: {self.metrics['val_mrr'][-1]:.3f}")
            logger.info(f"Best Val MRR: {max(self.metrics['val_mrr']):.3f}")
        
        if self.metrics.get("val_recall@5"):
            logger.info(f"Final Val Recall@5: {self.metrics['val_recall@5'][-1]:.3f}")
            logger.info(f"Best Val Recall@5: {max(self.metrics['val_recall@5']):.3f}")
        
        if self.metrics.get("val_precision@1"):
            logger.info(f"Final Val Precision@1: {self.metrics['val_precision@1'][-1]:.3f}")
            logger.info(f"Best Val Precision@1: {max(self.metrics['val_precision@1']):.3f}")
        
        logger.info("="*60 + "\n")


def main(args):
    """Main training pipeline."""
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load and process data
    processor = DataProcessor()
    dataset = processor.load_dataset(args.data)
    
    positives = [(x.get("question"), x.get("exp")) for x in dataset]
    
    # IMPROVED: Build multiple types of negatives
    logger.info("Building comprehensive negative set...")
    easy_negatives = processor.build_easy_negatives(dataset)
    hard_negatives = processor.build_hard_negatives(dataset)
    medical_hard_negatives = processor.build_medical_hard_negatives(dataset)
    contradiction_negatives = processor.build_contradiction_negatives(dataset)
    
    # Combine with strategic weighting (medical hard are most important!)
    negatives = (
        easy_negatives * 1 +  # 1x easy (baseline)
        hard_negatives * 2 +  # 2x hard (BM25 challenging)
        medical_hard_negatives * 3 +  # 3x medical hard (domain-specific!)
        contradiction_negatives * 2  # 2x contradictions (hallucination prevention!)
    )
    
    logger.info(f"Total weighted negatives: {len(negatives)}")
    
    # IMPROVED: Use augmentation if enabled
    if args.use_augmentation:
        triplets = create_triplets_with_augmentation(positives, negatives)
    else:
        triplets = processor.create_triplets(positives, negatives)
    
    # Train/val split
    split = int(0.9 * len(triplets))
    train_data = triplets[:split]
    val_data = triplets[split:]
    
    # Use subset if specified
    if args.use_subset:
        train_data = train_data[:len(train_data) // 10]
        val_data = val_data[:len(val_data) // 10]
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Model setup
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name)
    model = RankingScorer(encoder, embedding_dim=128).to(device)
    
    # Data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch_triplet(batch, tokenizer, args.max_len)
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch_triplet(batch, tokenizer, args.max_len)
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, tokenizer, device, args.output_dir)
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"Using loss: {'MNR' if args.use_mnr else 'Triplet'}")
    
    for epoch in range(args.epochs):
        # Choose training method
        if args.use_mnr:
            train_loss = trainer.train_epoch_mnr(
                train_loader, optimizer, scaler, epoch, 
                temperature=args.temperature
            )
        else:
            train_loss = trainer.train_epoch(
                train_loader, optimizer, scaler, epoch, 
                margin=args.margin
            )
        
        # Choose evaluation method
        if args.comprehensive_eval:
            results = trainer.evaluate_comprehensive(val_loader)
        else:
            val_acc, val_mrr = trainer.evaluate(val_loader)
    
    # Save model
    trainer.save_model(args.checkpoint_name)
    trainer.print_metrics_summary()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EBM Ranking Scorer model")
    parser.add_argument("--data", default="data/train.json", help="Path to training data")
    parser.add_argument("--model_name", default="distilbert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.5, help="Triplet loss margin")
    parser.add_argument("--max_len", type=int, default=128, help="Max token length")
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--checkpoint_name", default="ranking_scorer.ckpt", help="Checkpoint filename")
    parser.add_argument("--use_subset", action="store_true", help="Use 10% subset for fast prototyping")
    
    # Advanced training options
    parser.add_argument("--use_mnr", action="store_true", help="Use Multiple Negatives Ranking loss (recommended!)")
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for MNR loss")
    parser.add_argument("--use_augmentation", action="store_true", help="Use data augmentation (3x more data)")
    parser.add_argument("--comprehensive_eval", action="store_true", help="Use comprehensive evaluation metrics")
    
    args = parser.parse_args()
    main(args)
