# nert/models/contrastive/contrastive_retrieval.py
"""Updated contrastive model for retrieval and confidence calibration."""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
import csv
from collections import Counter
import spacy
from models.embedding_cache import get_encoder


class ContrastiveRetrievalModel(nn.Module):
    """Contrastive model for similarity-based retrieval and confidence estimation."""

    def __init__(self, base_encoder: str = 'all-MiniLM-L6-v2', projection_dim: int = 128):
        super().__init__()
        self.encoder = get_encoder()  # Use singleton encoder (loaded once, reused)
        self.input_dim = self.encoder.get_sentence_embedding_dimension()
        
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # For confidence calibration
        self.temperature = 1.0  # Will be learned during calibration
        self.calibrator = None  # Will hold isotonic regression model
        
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode texts with projection."""
        embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        return embeddings


class SafetyConfidenceEstimator:
    """Main class for retrieval-based confidence estimation."""

    def __init__(self, model_path: str = "models/trained_encoder.pt",
                 embeddings_path: str = "models/training_embeddings.pkl"):
        """Initialize with trained model and embeddings."""
        self.model = ContrastiveRetrievalModel()

        current_file = Path(__file__).resolve()

        for i in range(5):
            potential_root = current_file.parents[i]
            potential_model_path = potential_root / "models" / "trained_encoder.pt"
            potential_embed_path = potential_root / "models" / "training_embeddings.pkl"

            if potential_model_path.exists():
                model_path = str(potential_model_path)
                print(f"Found model at: {model_path}")
                break

            alt_model_path = potential_root / "trained_encoder.pt"
            if alt_model_path.exists():
                model_path = str(alt_model_path)
                print(f"Found model at: {model_path}")
                break
        else:
            print(f"WARNING: Could not find model file. Searched up to: {current_file.parents[4]}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Looking for: models/trained_encoder.pt")

        for i in range(5):
            potential_root = current_file.parents[i]
            potential_embed_path = potential_root / "models" / "training_embeddings.pkl"

            if potential_embed_path.exists():
                embeddings_path = str(potential_embed_path)
                print(f"Found embeddings at: {embeddings_path}")
                break

            alt_embed_path = potential_root / "training_embeddings.pkl"
            if alt_embed_path.exists():
                embeddings_path = str(alt_embed_path)
                print(f"Found embeddings at: {embeddings_path}")
                break

        self.load_model(model_path)
        self.load_training_embeddings(embeddings_path)
        self.calibrator = None

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        self.zero_shot_classifier = None

        self.danger_objects = {}
        self.load_danger_objects()
        
    def load_model(self, path: str):
        """Load trained contrastive model."""
        if Path(path).exists():
            checkpoint = torch.load(path, map_location='cpu')
            if 'projection' in checkpoint:
                self.model.projection.load_state_dict(checkpoint['projection'])
            self.safe_centroid = checkpoint.get('safe_centroid')
            self.unsafe_centroid = checkpoint.get('unsafe_centroid')
        else:
            print(f"Warning: Model not found at {path}, using base encoder only")
            self.safe_centroid = None
            self.unsafe_centroid = None
            
    def load_training_embeddings(self, path: str):
        """Load pre-computed training embeddings with optimized tensor operations."""
        if Path(path).exists():
            try:
                with open(path, 'rb') as f:
                    self.training_data = pickle.load(f)
                    embeddings = []
                    for item in self.training_data:
                        if isinstance(item['embedding'], np.ndarray):
                            item['embedding'] = torch.from_numpy(item['embedding']).float()
                        embeddings.append(item['embedding'])

                    if embeddings:
                        self.embedding_matrix = torch.stack(embeddings)
                    else:
                        self.embedding_matrix = None
            except Exception as e:
                error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
                print(f"Warning: Failed to load training embeddings: {error_msg}")
                self.training_data = []
                self.embedding_matrix = None
        else:
            print(f"Warning: Training embeddings not found at {path}")
            self.training_data = []
            self.embedding_matrix = None

    def load_danger_objects(self):
        """Load pre-computed danger objects from training."""
        current_file = Path(__file__).resolve()

        for i in range(5):
            potential_root = current_file.parents[i]
            danger_path = potential_root / "models" / "danger_objects.pkl"

            if danger_path.exists():
                try:
                    with open(danger_path, 'rb') as f:
                        self.danger_objects = pickle.load(f)
                    print(f"Loaded {len(self.danger_objects)} pre-computed danger objects")
                    if len(self.danger_objects) > 0:
                        top_dangers = sorted(self.danger_objects.items(), key=lambda x: x[1], reverse=True)[:10]
                        try:
                            danger_list = [f'{obj}({ratio:.2f})' for obj, ratio in top_dangers]
                            print(f"Top danger objects: {danger_list}")
                        except UnicodeEncodeError:
                            print(f"Top danger objects loaded (display suppressed due to encoding)")
                    return
                except Exception as e:
                    error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
                    print(f"Warning: Failed to load danger objects: {error_msg}")
                    self.danger_objects = {}
                    return

        print("Warning: Pre-computed danger objects not found. Run training to generate them.")
        self.danger_objects = {}
    
    def extract_danger_features_data_driven(self, prompt: str) -> Dict:
        """Extract danger features using learned patterns from training data."""
        if not self.nlp or not self.danger_objects:
            return {'danger_score': 0.0, 'detected_objects': [], 'detected_entities': [], 'detected_actions': []}

        doc = self.nlp(prompt)

        detected_objects = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        detected_entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]
        detected_actions = [token.text.lower() for token in doc if token.pos_ == 'VERB']

        danger_score = 0.0
        matched_objects = []

        for obj in detected_objects:
            if obj in self.danger_objects:
                ratio = self.danger_objects[obj]
                danger_score = max(danger_score, ratio)
                if ratio >= 0.7:
                    matched_objects.append(obj)

        vulnerable_keywords = {'child', 'children', 'kid', 'kids', 'baby', 'infant', 'toddler',
                              'person', 'people', 'patient', 'elderly', 'senior', 'minor', 'youth'}
        vulnerable_entities = [ent for ent in detected_entities if ent in vulnerable_keywords]
        vulnerable_entities.extend([obj for obj in detected_objects if obj in vulnerable_keywords])
        vulnerable_entities = list(dict.fromkeys(vulnerable_entities))

        return {
            'danger_score': danger_score,
            'detected_objects': matched_objects,  
            'detected_entities': vulnerable_entities[:3] if matched_objects else [],
            'detected_actions': detected_actions[:3] if matched_objects else [],
            'danger_detected': danger_score >= 0.7
        }

    def extract_danger_features_zero_shot(self, prompt: str) -> Dict:
        """Zero-shot classification using BART."""
        # COMMENTED OUT: BART loading disabled for performance testing
        # if not self.nlp:
        #     return {'danger_score': 0.0, 'detected_objects': [], 'detected_entities': [], 'detected_actions': [], 'primary_hazard': None}
        #
        # # Lazy load zero-shot classifier
        # if self.zero_shot_classifier is None:
        #     try:
        #         from transformers import pipeline
        #         self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
        #     except Exception as e:
        #         print(f"Warning: Could not load zero-shot classifier: {e}")
        #         return {'danger_score': 0.0, 'detected_objects': [], 'detected_entities': [], 'detected_actions': [], 'primary_hazard': None}

        # Return empty results without loading BART
        return {'danger_score': 0.0, 'detected_objects': [], 'detected_entities': [], 'detected_actions': [], 'primary_hazard': None}

        doc = self.nlp(prompt)

        # Extract entities
        detected_objects = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        detected_entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]
        detected_actions = [token.text.lower() for token in doc if token.pos_ == 'VERB']

        # Filter out non-dangerous generic objects
        generic_objects = {'scene', 'context', 'kitchen', 'area', 'space', 'room', 'location', 'place', 'environment'}
        detected_objects = [obj for obj in detected_objects if obj not in generic_objects]

        # Find vulnerable people in BOTH entities and objects (spaCy might not tag "child" as PERSON)
        vulnerable_keywords = {'child', 'children', 'kid', 'kids', 'baby', 'infant', 'toddler',
                              'person', 'people', 'patient', 'elderly', 'senior', 'minor', 'youth'}
        vulnerable_entities = [ent for ent in detected_entities if ent in vulnerable_keywords]
        vulnerable_entities.extend([obj for obj in detected_objects if obj in vulnerable_keywords])
        vulnerable_entities = list(dict.fromkeys(vulnerable_entities))  # Remove duplicates while preserving order

        if not detected_objects:
            return {'danger_score': 0.0, 'detected_objects': [], 'detected_entities': [], 'detected_actions': [], 'primary_hazard': None}

        # Classify primary object
        primary_object = detected_objects[0]

        try:
            hypothesis = f"This is a {primary_object}"
            result = self.zero_shot_classifier(hypothesis, candidate_labels=[
                "sharp or cutting object",
                "heavy or crushing object",
                "toxic or poisonous substance",
                "flammable or explosive material",
                "medication or drug",
                "electrical or high voltage equipment",
                "extreme temperature item"
            ], multi_label=True)

            # Get highest scoring hazard
            max_score = max(result['scores'])
            max_label = result['labels'][result['scores'].index(max_score)]

            # Assess vulnerability if entities detected
            vulnerability_score = 0.0
            if detected_entities:
                entity = detected_entities[0]
                vulnerability_query = f"A {entity} is vulnerable to a {primary_object}"
                vuln_result = self.zero_shot_classifier(vulnerability_query,
                    candidate_labels=["high risk", "moderate risk", "low risk"])

                if vuln_result['labels'][0] == "high risk":
                    vulnerability_score = vuln_result['scores'][0]

            # Combine object danger and vulnerability
            danger_score = max_score
            if vulnerability_score > 0:
                danger_score = min(1.0, (max_score + vulnerability_score) / 2)

            # Only return the primary dangerous object if it scored high
            dangerous_objects = [primary_object] if max_score > 0.5 else []

            return {
                'danger_score': danger_score,
                'detected_objects': dangerous_objects,
                'detected_entities': vulnerable_entities[:3],
                'detected_actions': detected_actions[:3],
                'primary_hazard': max_label if max_score > 0.5 else None
            }

        except Exception as e:
            print(f"Warning: Zero-shot classification failed: {e}")
            return {'danger_score': 0.0, 'detected_objects': detected_objects[:3], 'detected_entities': detected_entities[:3], 'detected_actions': detected_actions[:3], 'primary_hazard': None}

    def extract_danger_features_hybrid(self, prompt: str) -> Dict:
        """Combine data-driven and zero-shot approaches."""
        data_result = self.extract_danger_features_data_driven(prompt)
        zero_result = self.extract_danger_features_zero_shot(prompt)

        combined_score = max(data_result['danger_score'], zero_result['danger_score'])

        if data_result['detected_objects']:
            display_objects = data_result['detected_objects']
            display_entities = data_result['detected_entities']
            display_actions = data_result['detected_actions']
        else:
            display_objects = zero_result['detected_objects'][:3]
            display_entities = zero_result['detected_entities'][:3]
            display_actions = zero_result['detected_actions'][:3]

        return {
            'danger_detected': combined_score >= 0.7, 
            'danger_score': combined_score,
            'detected_objects': display_objects,
            'detected_entities': display_entities,
            'detected_actions': display_actions,
            'data_driven_score': data_result['danger_score'],
            'zero_shot_score': zero_result['danger_score'],
            'primary_hazard': zero_result.get('primary_hazard')
        }

    def retrieve_similar_examples(self, task: str, k: int = 10) -> List[Dict]:
        """
        Retrieve k most similar examples from training data using vectorized operations.
        Returns list of dicts with 'task', 'label', 'similarity' keys.
        """
        if not self.training_data or self.embedding_matrix is None:
            return []

        query_embedding = self.model.encode_text([task])[0]

        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.embedding_matrix,
            dim=1
        )

        top_k_indices = torch.topk(similarities, k=min(k, len(similarities)), largest=True).indices

        results = []
        for idx in top_k_indices:
            idx = idx.item()
            results.append({
                'task': self.training_data[idx]['task'],
                'label': self.training_data[idx]['label'],
                'similarity': similarities[idx].item()
            })

        return results

    def compute_pattern_overlap(self, danger1: Dict, danger2: Dict) -> float:
        """
        Compute overlap between two danger patterns.
        Returns 0-1 score.
        """
        if not danger1.get('danger_detected') or not danger2.get('danger_detected'):
            return 0.0

        score = 0.0

        obj1 = set(danger1.get('detected_objects', []))
        obj2 = set(danger2.get('detected_objects', []))
        if obj1 and obj2:
            score += len(obj1 & obj2) / max(len(obj1), len(obj2)) * 0.4

        if danger1.get('primary_hazard') and danger2.get('primary_hazard'):
            if danger1['primary_hazard'] == danger2['primary_hazard']:
                score += 0.3  

        ent1 = set(danger1.get('detected_entities', []))
        ent2 = set(danger2.get('detected_entities', []))
        if ent1 and ent2:
            score += len(ent1 & ent2) / max(len(ent1), len(ent2)) * 0.3

        act1 = set(danger1.get('detected_actions', []))
        act2 = set(danger2.get('detected_actions', []))
        if act1 and act2:
            score += len(act1 & act2) / max(len(act1), len(act2)) * 0.3

        return min(score, 1.0)

    def semantic_search(self, query_embedding: torch.Tensor, pool: List[Dict], k: int) -> List[Dict]:
        """Standard cosine similarity search within a pool."""
        results = []

        for example in pool:
            sim = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                example['embedding'].unsqueeze(0),
                dim=1
            ).item()
            results.append({
                'task': example['task'],
                'label': example['label'],
                'similarity': sim
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]

    def danger_aware_search(self, task: str, danger_analysis: Dict, pool: List[Dict], k: int) -> List[Dict]:
        """
        Enhanced similarity combining semantic and danger pattern matching.
        Uses PRE-COMPUTED danger features for fast performance.
        """
        query_embedding = self.model.encode_text([task])[0]
        results = []

        for example in pool:
            semantic_sim = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                example['embedding'].unsqueeze(0),
                dim=1
            ).item()

            example_danger = example.get('danger_features', {})

            danger_sim = 0.0
            if example_danger and danger_analysis.get('danger_detected'):
                danger_sim = self.compute_pattern_overlap(danger_analysis, example_danger)

            combined = 0.6 * semantic_sim + 0.4 * danger_sim

            results.append({
                'task': example['task'],
                'label': example['label'],
                'similarity': combined
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]

    def multi_retrieval(self, task: str, symbolic_decision: str, danger_analysis: Dict, k: int = 10) -> Dict:
        """
        Pool-based retrieval with optional danger pattern matching.

        Args:
            task: The task to evaluate
            symbolic_decision: 'ACCEPT' or 'REJECT'
            danger_analysis: Output from extract_danger_features_hybrid
            k: Number of primary examples to retrieve

        Returns:
            Dict with 'primary_examples', optional 'contrasting_examples', and 'display_mode'
        """
        if not self.training_data:
            return {'primary_examples': [], 'display_mode': 'standard'}

        query_embedding = self.model.encode_text([task])[0]

        unsafe_pool = [ex for ex in self.training_data if ex['label'] == 'unsafe']
        safe_pool = [ex for ex in self.training_data if ex['label'] == 'safe']

        if symbolic_decision == 'REJECT':
            if danger_analysis.get('danger_score', 0) >= 0.7:
                # High danger: use danger-aware search
                primary = self.danger_aware_search(task, danger_analysis, unsafe_pool, k)

                # Also get contrasting safe examples
                contrasting = self.semantic_search(query_embedding, safe_pool, k=5)

                return {
                    'primary_examples': primary,
                    'contrasting_examples': contrasting,
                    'display_mode': 'danger_with_alternatives'
                }
            else:
                # Low/no danger: standard semantic search in unsafe pool
                primary = self.semantic_search(query_embedding, unsafe_pool, k)
                return {
                    'primary_examples': primary,
                    'display_mode': 'standard'
                }
        else:
            # ACCEPT: search safe pool only
            primary = self.semantic_search(query_embedding, safe_pool, k)
            return {
                'primary_examples': primary,
                'display_mode': 'standard'
            }

    def estimate_confidence(self, task: str, symbolic_decision: str,
                        k_neighbors: int = 10, temperature: float = 2.0) -> Dict:
        """
        Estimate confidence based on agreement with similar examples.
        USES DUAL RETRIEVAL APPROACH:

        1. Retrieve k=10 from ENTIRE dataset → calculate metrics (support_ratio, weighted_confidence)
        2. Retrieve k=5 from POOLS → for UI display only

        Args:
            task: The task to evaluate
            symbolic_decision: The decision from symbolic checker ('ACCEPT' or 'REJECT')
            k_neighbors: Number of neighbors for metrics calculation (default: 10)
            temperature: Temperature for scaling confidence (higher = more moderate)

        Returns:
            Dict with 'confidence', 'danger_analysis', 'nearest_neighbors', 'contrasting_examples'
        """
        danger_analysis = self.extract_danger_features_hybrid(task)

        all_data_neighbors = self.retrieve_similar_examples(task, k=10)

        # Retrieve k=5 from pools for display (better UX)
        retrieval_result = self.multi_retrieval(task, symbolic_decision, danger_analysis, k=5)

        primary_examples = retrieval_result.get('primary_examples', [])
        contrasting_examples = retrieval_result.get('contrasting_examples', [])

        if not all_data_neighbors:
            return {
                'confidence': 0.5,
                'support_ratio': 0.5,
                'danger_analysis': danger_analysis,
                'nearest_neighbors': [],
                'contrasting_examples': [],
                'uncertainty_reason': 'No training data available'
            }

        # Calculate metrics from ALL DATA (k=10) - NOT from pools
        expected_label = 'safe' if symbolic_decision == 'ACCEPT' else 'unsafe'

        # Support ratio: proportion of k=10 neighbors that agree with symbolic decision
        agreeing_neighbors = sum(
            1 for ex in all_data_neighbors
            if ex['label'] == expected_label
        )
        support_ratio = agreeing_neighbors / len(all_data_neighbors)

        # Weighted confidence: similarity-weighted agreement from k=10 neighbors
        # Closer examples (higher similarity) contribute more to confidence
        weighted_agreement = sum(
            ex['similarity'] for ex in all_data_neighbors
            if ex['label'] == expected_label
        )
        total_weight = sum(ex['similarity'] for ex in all_data_neighbors)
        weighted_confidence = weighted_agreement / total_weight if total_weight > 0 else 0.5

        # Apply temperature scaling to prevent extreme confidences
        if temperature > 0:
            # Convert to logit, scale by temperature, then back to probability
            # This prevents extreme 0 and 1 values
            logit = (weighted_confidence - 0.5) * temperature
            scaled_confidence = 1 / (1 + np.exp(-logit))
        else:
            scaled_confidence = weighted_confidence

        # Apply calibration if available
        if self.calibrator is not None:
            # Use isotonic regression calibration on the scaled confidence
            raw_conf = np.array([[scaled_confidence]])
            calibrated_conf = self.calibrator.predict(raw_conf)[0]
        else:
            calibrated_conf = scaled_confidence

        # Determine uncertainty reason based on metrics from all_data_neighbors
        uncertainty_reason = None
        if support_ratio < 0.3:
            uncertainty_reason = "Strong disagreement with similar examples"
        elif support_ratio < 0.7:
            uncertainty_reason = "Mixed signals from similar examples"
        elif all_data_neighbors[0]['similarity'] < 0.5:
            uncertainty_reason = "No highly similar examples found"

        # STEP 4: Return results
        # NOTE: Metrics (confidence, support_ratio) calculated from k=10 ALL data
        #       Display (nearest_neighbors) uses k=5 from POOLS for better UX

        return {
            'confidence': calibrated_conf,  # From all_data_neighbors (k=10)
            'support_ratio': support_ratio,  # From all_data_neighbors (k=10)
            'raw_confidence': weighted_confidence,  # From all_data_neighbors
            'scaled_confidence': scaled_confidence,  # From all_data_neighbors
            'danger_analysis': danger_analysis,
            'nearest_neighbors': primary_examples,  # For UI display only (k=5 from pools)
            'contrasting_examples': contrasting_examples,  # For UI display (safe alternatives)
            'uncertainty_reason': uncertainty_reason
        }
    
    def calibrate_confidence(self, validation_data: List[Tuple[str, str, str]]):
        """
        Calibrate confidence scores using validation data.
        
        Args:
            validation_data: List of (task, symbolic_decision, ground_truth) tuples
        """
        print("Calibrating confidence scores...")
        
        raw_confidences = []
        is_correct = []
        
        for task, symbolic_decision, ground_truth in validation_data:
            result = self.estimate_confidence(task, symbolic_decision)
            raw_conf = result['weighted_confidence']
            
            predicted_label = 'safe' if symbolic_decision == 'ACCEPT' else 'unsafe'
            correct = (predicted_label == ground_truth)
            
            raw_confidences.append(raw_conf)
            is_correct.append(1 if correct else 0)
        
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(raw_confidences, is_correct)
        
        print(f"Calibration complete using {len(validation_data)} examples")
    
    def get_enhanced_context(self, task: str, k: int = 5) -> str:
        """
        Generate enhanced context string with similar examples for LLM prompting.
        
        Args:
            task: The task to evaluate
            k: Number of examples to include
            
        Returns:
            Context string to include in prompt
        """
        similar_examples = self.retrieve_similar_examples(task, k)
        
        if not similar_examples:
            return ""
        
        context = "Similar past tasks and their safety outcomes:\n"
        for i, ex in enumerate(similar_examples, 1):
            context += f"{i}. Task: '{ex['task']}' - Classification: {ex['label']} (similarity: {ex['similarity']:.2f})\n"
        
        return context


class UpdatedNeuralConfidenceModel:
    """Updated neural confidence model using retrieval."""
    
    def __init__(self, model_path: str = "models/trained_encoder.pt"):
        self.confidence_estimator = SafetyConfidenceEstimator(model_path)
        
    def predict(self, task: str, symbolic_decision: str = None) -> Dict:
        """Get confidence and nearest neighbors."""
        if symbolic_decision:
            result = self.confidence_estimator.estimate_confidence(
                task, symbolic_decision, k_neighbors=10
            )
        else:
            similar = self.confidence_estimator.retrieve_similar_examples(task, k=10)
            result = {
                'confidence': 0.5, 
                'similar_examples': similar
            }
        
        return {
            'confidence': result['confidence'],
            'nearest_neighbors': [
                {
                    'task': ex['task'],
                    'label': ex['label'],
                    'distance': 1 - ex['similarity']
                }
                for ex in result.get('similar_examples', [])[:5]
            ],
            'uncertainty_reason': result.get('uncertainty_reason')
        }