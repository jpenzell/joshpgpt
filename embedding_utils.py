import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import hashlib

@dataclass
class EmbeddingMetadata:
    model_name: str
    model_version: str
    dimensions: int
    created_at: str
    content_hash: str
    content_type: str  # text, image, audio, pdf, etc.
    processing_params: Dict
    quality_metrics: Dict

class EmbeddingManager:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.version_history_file = os.path.join(cache_dir, "embedding_versions.json")
        self.quality_thresholds = {
            "min_nonzero_components": 0.1,  # At least 10% of components should be non-zero
            "max_magnitude_threshold": 100,  # Maximum allowed vector magnitude
            "min_entropy": 0.5  # Minimum information entropy
        }
        os.makedirs(cache_dir, exist_ok=True)
        self._load_version_history()

    def _load_version_history(self):
        if os.path.exists(self.version_history_file):
            with open(self.version_history_file, 'r') as f:
                self.version_history = json.load(f)
        else:
            self.version_history = {}

    def _save_version_history(self):
        with open(self.version_history_file, 'w') as f:
            json.dump(self.version_history, f, indent=2)

    def compute_content_hash(self, content: Union[str, bytes]) -> str:
        """Generate a unique hash for the content"""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()

    def validate_embedding_quality(self, embedding: List[float]) -> Tuple[bool, Dict]:
        """Validate embedding quality using multiple metrics"""
        embedding_array = np.array(embedding)
        
        # Calculate quality metrics
        nonzero_ratio = np.count_nonzero(embedding_array) / len(embedding_array)
        magnitude = np.linalg.norm(embedding_array)
        entropy = -np.sum(np.square(embedding_array) * np.log(np.square(embedding_array) + 1e-10))
        
        # Quality checks
        quality_metrics = {
            "nonzero_ratio": nonzero_ratio,
            "magnitude": magnitude,
            "entropy": entropy
        }
        
        # Validation results
        is_valid = (
            nonzero_ratio >= self.quality_thresholds["min_nonzero_components"] and
            magnitude <= self.quality_thresholds["max_magnitude_threshold"] and
            entropy >= self.quality_thresholds["min_entropy"]
        )
        
        return is_valid, quality_metrics

    def create_embedding_metadata(
        self,
        content: Union[str, bytes],
        model_name: str,
        dimensions: int,
        content_type: str,
        processing_params: Dict,
        quality_metrics: Dict
    ) -> EmbeddingMetadata:
        """Create metadata for an embedding"""
        return EmbeddingMetadata(
            model_name=model_name,
            model_version=self.version_history.get(model_name, {}).get("latest", "unknown"),
            dimensions=dimensions,
            created_at=datetime.now().isoformat(),
            content_hash=self.compute_content_hash(content),
            content_type=content_type,
            processing_params=processing_params,
            quality_metrics=quality_metrics
        )

    def should_recompute_embedding(
        self,
        content_hash: str,
        model_name: str,
        current_version: str
    ) -> bool:
        """Determine if an embedding should be recomputed based on version history"""
        if content_hash not in self.version_history:
            return True
        
        content_version = self.version_history[content_hash].get("model_version")
        if not content_version:
            return True
        
        # Compare versions and check if recomputation is needed
        return self._compare_versions(content_version, current_version) < 0

    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings"""
        v1_parts = [int(x) for x in version1.split(".")]
        v2_parts = [int(x) for x in version2.split(".")]
        
        for i in range(max(len(v1_parts), len(v2_parts))):
            v1 = v1_parts[i] if i < len(v1_parts) else 0
            v2 = v2_parts[i] if i < len(v2_parts) else 0
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
        return 0

    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector"""
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            return (embedding_array / norm).tolist()
        return embedding

    def combine_embeddings(
        self,
        embeddings: List[List[float]],
        weights: Optional[List[float]] = None
    ) -> List[float]:
        """Combine multiple embeddings with optional weights"""
        if not embeddings:
            return []
        
        if weights is None:
            weights = [1.0] * len(embeddings)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Weighted average of embeddings
        combined = np.average(embeddings, weights=weights, axis=0)
        
        # Normalize the result
        return self.normalize_embedding(combined.tolist()) 