# nert/models/contrastive/train_contrastive.py
"""Train contrastive learning model for safety classification."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import pickle
from pathlib import Path
from collections import Counter
import spacy


class SafetyDataset(Dataset):
    """Dataset for contrastive learning."""
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, encoding='utf-8')
        self.tasks = self.df['task_prompt'].tolist()
        self.labels = self.df['safety_classification'].map({'safe': 1, 'unsafe': 0}).tolist()
        
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        return self.tasks[idx], self.labels[idx]
    
    def get_triplets(self) -> List[Tuple[str, str, str]]:
        """Generate triplets for contrastive learning."""
        triplets = []
        
        safe_tasks = self.df[self.df['safety_classification'] == 'safe']['task_prompt'].tolist()
        unsafe_tasks = self.df[self.df['safety_classification'] == 'unsafe']['task_prompt'].tolist()
        
        for i, (task, label) in enumerate(zip(self.tasks, self.labels)):
            if label == 1: 
                positive = np.random.choice([t for t in safe_tasks if t != task])
                negative = np.random.choice(unsafe_tasks)
            else: 
                positive = np.random.choice([t for t in unsafe_tasks if t != task])
                negative = np.random.choice(safe_tasks)
            
            triplets.append((task, positive, negative))
        
        return triplets


class ContrastiveModel(nn.Module):
    """Contrastive learning model for safety classification."""
    
    def __init__(self, base_encoder: str = 'all-MiniLM-L6-v2', projection_dim: int = 128):
        super().__init__()
        self.encoder = SentenceTransformer(base_encoder)
        self.input_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
            nn.LayerNorm(projection_dim)
        )
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts and apply projection."""
        embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        embeddings = embeddings.clone().detach().requires_grad_(True)        
        return self.projection(embeddings)
    
    def compute_triplet_loss(self, anchor: torch.Tensor, positive: torch.Tensor, 
                            negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        """Compute triplet loss."""
        pos_dist = torch.norm(anchor - positive, dim=1)
        neg_dist = torch.norm(anchor - negative, dim=1)
        loss = torch.relu(pos_dist - neg_dist + margin)
        return loss.mean()


def train_contrastive_model(dataset_path: str, output_path: str, epochs: int = 10):
    """Train the contrastive model."""
    import sys
    if sys.platform == 'win32':
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass 

    print("=" * 60, flush=True)
    print("CONTRASTIVE MODEL TRAINING", flush=True)
    print("=" * 60, flush=True)

    print(f"\nLoading dataset from: {dataset_path}", flush=True)
    dataset = SafetyDataset(dataset_path)
    print(f"[OK] Loaded {len(dataset)} examples", flush=True)

    triplets = dataset.get_triplets()
    print(f"[OK] Generated {len(triplets)} triplets for training\n", flush=True)

    print("Initializing contrastive model...", flush=True)
    model = ContrastiveModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("[OK] Model initialized\n", flush=True)

    print("Starting training...", flush=True)
    print("-" * 60, flush=True)
    batch_size = 32
    num_batches = len(triplets) // batch_size

    for epoch in range(epochs):
        total_loss = 0

        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i+batch_size]

            anchors = [t[0] for t in batch]
            positives = [t[1] for t in batch]
            negatives = [t[2] for t in batch]

            anchor_emb = model(anchors)
            positive_emb = model(positives)
            negative_emb = model(negatives)

            loss = model.compute_triplet_loss(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            batch_num = i // batch_size + 1
            if batch_num % 100 == 0:
                current_avg_loss = total_loss / (batch_num * batch_size)
                print(f"  Epoch {epoch+1}/{epochs} - Batch {batch_num}/{num_batches} - Avg Loss: {current_avg_loss:.4f}", flush=True)

        avg_loss = total_loss / len(triplets) * batch_size
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", flush=True)

    print("-" * 60, flush=True)
    print("[OK] Training complete!\n", flush=True)

    print("Saving model...", flush=True)
    save_model(model, dataset, output_path)
    print(f"[OK] Model saved to {output_path}", flush=True)
    print("=" * 60, flush=True)


def discover_danger_patterns(tasks: List[str], labels: List[str], nlp) -> Dict[str, float]:
    """
    Discover danger-associated objects from training data.
    """
    print("Discovering danger patterns from training data...", flush=True)
    print("-" * 60, flush=True)

    unsafe_objects = Counter()
    safe_objects = Counter()

    # Generic words to filter out (not inherently dangerous)
    GENERIC_WORDS = {
        'zone', 'space', 'area', 'materials', 'items', 'things', 'objects',
        'procedures', 'processes', 'activities', 'actions', 'tasks',
        'people', 'person', 'individual', 'users', 'workers'
    }

   
    print("Analyzing task objects using noun chunks...", flush=True)
    skipped_count = 0
    for i, (task, label) in enumerate(zip(tasks, labels)):
        try:
            doc = nlp(task)

            objects = set()

            single_nouns = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN']]
            objects.update(single_nouns)

            noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 2]
            objects.update(noun_chunks)

            objects = {obj for obj in objects if obj not in GENERIC_WORDS and len(obj) > 2}

            if label == 'unsafe' or label == 0:
                unsafe_objects.update(objects)
            else:
                safe_objects.update(objects)
        except Exception as e:
            skipped_count += 1
            continue

        if (i + 1) % 2000 == 0:
            print(f"  Analyzed {i + 1}/{len(tasks)} tasks...", flush=True)

    print(f"[OK] Analysis complete", flush=True)
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} problematic tasks", flush=True)
    print(f"  Found {len(unsafe_objects)} unique objects in unsafe tasks", flush=True)
    print(f"  Found {len(safe_objects)} unique objects in safe tasks\n", flush=True)

    # Find objects with high unsafe ratio
    print("Filtering danger-associated objects...", flush=True)
    print("  Criteria: >=2 occurrences AND >=40% appear in unsafe tasks", flush=True)
    print("  Filtering out generic words (zone, space, materials, etc.)", flush=True)

    danger_objects = {}
    for obj, unsafe_count in unsafe_objects.items():
        safe_count = safe_objects.get(obj, 0)
        total = unsafe_count + safe_count

        if total >= 2: 
            danger_ratio = unsafe_count / total
            if danger_ratio >= 0.4: 
                danger_objects[obj] = danger_ratio

    print(f"\n[OK] Discovered {len(danger_objects)} danger-associated objects", flush=True)

    if len(danger_objects) > 0:
        top_dangers = sorted(danger_objects.items(), key=lambda x: x[1], reverse=True)[:15]
        print(f"\nTop 15 danger objects:", flush=True)
        for obj, ratio in top_dangers:
            print(f"  - '{obj}': {ratio*100:.2f}% unsafe", flush=True)
    else:
        print("[WARNING] No danger objects found! This may indicate an issue.", flush=True)

    print("-" * 60, flush=True)

    return danger_objects


def extract_danger_features_data_driven(task: str, nlp, danger_objects: Dict[str, float]) -> Dict:
    """
    Extract danger features using learned patterns from training data.
    This is the fast version without zero-shot classification.
    """
    if not nlp:
        return {
            'danger_score': 0.0,
            'detected_objects': [],
            'detected_entities': [],
            'detected_actions': []
        }

    try:
        doc = nlp(task)

        detected_objects = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        detected_entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]
        detected_actions = [token.text.lower() for token in doc if token.pos_ == 'VERB']

        danger_score = 0.0
        matched_objects = []

        for obj in detected_objects:
            if obj in danger_objects:
                danger_score = max(danger_score, danger_objects[obj])
                matched_objects.append(obj)

        return {
            'danger_score': danger_score,
            'detected_objects': matched_objects if matched_objects else detected_objects[:3],
            'detected_entities': detected_entities[:3],
            'detected_actions': detected_actions[:3],
            'danger_detected': danger_score >= 0.7
        }
    except Exception as e:
        return {
            'danger_score': 0.0,
            'detected_objects': [],
            'detected_entities': [],
            'detected_actions': [],
            'danger_detected': False
        }


def save_model(model: ContrastiveModel, dataset: SafetyDataset, output_path: str):
    """Save trained model and compute cluster centroids."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_embeddings = []
    all_tasks = []
    all_labels = []
    
    with torch.no_grad():
        for task, label in zip(dataset.tasks, dataset.labels):
            embedding = model([task])[0]
            all_embeddings.append(embedding)
            all_tasks.append(task)
            all_labels.append(label)
    
    safe_embeddings = [emb for emb, label in zip(all_embeddings, all_labels) if label == 1]
    unsafe_embeddings = [emb for emb, label in zip(all_embeddings, all_labels) if label == 0]
    
    safe_centroid = torch.stack(safe_embeddings).mean(dim=0) if safe_embeddings else torch.zeros_like(all_embeddings[0])
    unsafe_centroid = torch.stack(unsafe_embeddings).mean(dim=0) if unsafe_embeddings else torch.zeros_like(all_embeddings[0])
    
    checkpoint = {
        'projection': model.projection.state_dict(),
        'safe_centroid': safe_centroid,
        'unsafe_centroid': unsafe_centroid
    }
    torch.save(checkpoint, output_path)
    
    print("\n" + "=" * 60, flush=True)
    print("DANGER FEATURE EXTRACTION", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading spaCy NLP model...", flush=True)
    try:
        nlp = spacy.load("en_core_web_sm")
        print("[OK] spaCy model loaded successfully\n", flush=True)
    except:
        print("[ERROR] WARNING: spaCy model not found. Skipping danger feature extraction.", flush=True)
        print("Install with: python -m spacy download en_core_web_sm\n", flush=True)
        nlp = None

    danger_objects = {}
    if nlp:
        string_labels = ['safe' if label == 1 else 'unsafe' for label in all_labels]
        danger_objects = discover_danger_patterns(all_tasks, string_labels, nlp)

    print("\nExtracting danger features for all examples...", flush=True)
    print("-" * 60, flush=True)
    training_data = []

    for i, (task, label, emb) in enumerate(zip(all_tasks, all_labels, all_embeddings)):
        item = {
            'task': task,
            'label': 'safe' if label == 1 else 'unsafe', 
            'embedding': emb.cpu().numpy()
        }

        if nlp:
            danger_features = extract_danger_features_data_driven(task, nlp, danger_objects)
            item['danger_features'] = danger_features

        training_data.append(item)

        if (i + 1) % 1000 == 0:
            percent = (i + 1) / len(all_tasks) * 100
            print(f"  Progress: {i + 1}/{len(all_tasks)} ({percent:.1f}%) examples processed...", flush=True)

    print("-" * 60, flush=True)
    print(f"[OK] Processed all {len(all_tasks)} examples\n", flush=True)

    print("Saving embeddings and danger features...", flush=True)
    with open(output_dir / 'training_embeddings.pkl', 'wb') as f:
        pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Saved {len(training_data)} training embeddings + danger features", flush=True)
    print(f"  Location: {output_dir / 'training_embeddings.pkl'}\n", flush=True)

    if danger_objects:
        with open(output_dir / 'danger_objects.pkl', 'wb') as f:
            pickle.dump(danger_objects, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[OK] Saved {len(danger_objects)} danger objects", flush=True)
        print(f"  Location: {output_dir / 'danger_objects.pkl'}", flush=True)

    print("=" * 60, flush=True)
    print("TRAINING COMPLETE!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../../data/training_1200.csv')
    parser.add_argument('--output', default='../trained_encoder.pt')
    parser.add_argument('--epochs', type=int, default=10)
    
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    output_path = Path(args.output).resolve()

    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        print(f"Please ensure training data exists at: {data_path}")
        exit(1)
    
    train_contrastive_model(args.data, args.output, args.epochs)