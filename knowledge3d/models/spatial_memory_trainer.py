#!/usr/bin/env python3
"""
Spatial Memory Trainer for K3D AGI Development
Trains AI models using sleep-compute optimized knowledge graphs
Implements multi-modal learning: text + spatial + temporal
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
import datetime
import random
from pathlib import Path

try:
    # Enforce containment for training on Debian-like systems
    from ..utils.env_guard import enforce_containment  # type: ignore
    enforce_containment("Spatial memory training")
except Exception:
    pass

class SpatialMemoryDataset(Dataset):
    """Dataset for spatial memory training using optimized knowledge graphs."""

    def __init__(self, data_file: str, embedding_model=None):
        """Load and prepare spatial memory training data."""
        with open(data_file, 'r') as f:
            self.nodes = json.load(f)

        # Filter for nodes with spatial data and embeddings
        self.spatial_nodes = [
            node for node in self.nodes
            if node.get('spatial', {}).get('position') and
            node.get('vectors', {}).get('original') and
            node['metadata'].get('confidence', 0) > 0.3
        ]

        print(f"Loaded {len(self.spatial_nodes)} nodes for spatial training")
        print(f"Compression levels: {set(n['compression']['level'] for n in self.spatial_nodes) if self.spatial_nodes else 'None'}")

        self.embedding_model = embedding_model
        self._build_relationships()

    def _build_relationships(self):
        """Build spatial and semantic relationship matrix."""
        self.position_matrix = []
        self.embedding_matrix = []
        self.relationships = []

        for i, node in enumerate(self.spatial_nodes):
            pos = node['spatial']['position']
            self.position_matrix.append([pos['x'], pos['y'], pos['z']])

            embedding = node['vectors']['original'][:768]  # Truncate for efficiency
            self.embedding_matrix.append(embedding)

        self.position_matrix = np.array(self.position_matrix)
        self.embedding_matrix = np.array(self.embedding_matrix)

        # Build spatial adjacency matrix
        self.spatial_distances = np.zeros((len(self.spatial_nodes), len(self.spatial_nodes)))
        for i in range(len(self.spatial_nodes)):
            for j in range(len(self.spatial_nodes)):
                if i != j:
                    pos_i = self.position_matrix[i]
                    pos_j = self.position_matrix[j]
                    self.spatial_distances[i,j] = np.linalg.norm(pos_i - pos_j)

        # Build semantic similarity matrix
        self.semantic_similarities = cosine_similarity(self.embedding_matrix)

        print(f"Built relationship matrices: {self.spatial_distances.shape}")

    def __len__(self):
        return len(self.spatial_nodes) * 10  # Generate multiple samples per node

    def __getitem__(self, idx):
        """Generate training sample with spatial reasoning task."""
        node_idx = idx % len(self.spatial_nodes)
        task_type = random.choice(['navigation', 'similarity', 'compression', 'temporal'])

        if task_type == 'navigation':
            return self._create_navigation_sample(node_idx)
        elif task_type == 'similarity':
            return self._create_similarity_sample(node_idx)
        elif task_type == 'compression':
            return self._create_compression_sample(node_idx)
        else:  # temporal
            return self._create_temporal_sample(node_idx)

    def _create_navigation_sample(self, node_idx):
        """Create sample for spatial navigation prediction."""
        node = self.spatial_nodes[node_idx]
        position = np.array([node['spatial']['position'][k] for k in ['x', 'y', 'z']])

        # Find nearby nodes
        distances = self.spatial_distances[node_idx]
        nearby_idx = np.argsort(distances)[1:4]  # 3 closest neighbors

        # Choose target
        target_idx = random.choice(nearby_idx)
        target_position = self.position_matrix[target_idx]
        target_distance = distances[target_idx]

        # Input: current embedding + position + context embeddings of nearby
        context_embeddings = [
            self.embedding_matrix[i] for i in nearby_idx
        ]

        input_features = np.concatenate([
            self.embedding_matrix[node_idx],
            position,
            *context_embeddings
        ])

        # Label: target position relative to current
        relative_position = target_position - position
        normalized_distance = target_distance / np.max(distances)

        return {
            'input': torch.FloatTensor(input_features),
            'task_type': 'navigation',
            'target_position': torch.FloatTensor(relative_position),
            'target_distance': torch.FloatTensor([normalized_distance]),
            'node_id': node['id']
        }

    def _create_similarity_sample(self, node_idx):
        """Create sample for semantic similarity prediction."""
        node = self.spatial_nodes[node_idx]

        # Find most and least similar nodes
        similarities = self.semantic_similarities[node_idx]
        sorted_idx = np.argsort(similarities)

        most_similar_idx = sorted_idx[-2]  # Second most similar (exclude self)
        least_similar_idx = sorted_idx[1]  # Second least similar

        input_features = np.concatenate([
            self.embedding_matrix[node_idx],
            self.embedding_matrix[most_similar_idx],
            self.embedding_matrix[least_similar_idx]
        ])

        # Label: classify which is more similar
        return {
            'input': torch.FloatTensor(input_features),
            'task_type': 'similarity',
            'similarity_label': torch.LongTensor([0]),  # First neighbor is more similar
            'node_id': node['id']
        }

    def _create_compression_sample(self, node_idx):
        """Create sample for compression decision making."""
        node = self.spatial_nodes[node_idx]

        compression_level = node['compression']['level']
        memory_strength = node['ai_native']['cognition']['memory_strength']
        utility_score = node['ai_native']['training']['utility_score']

        # Features for compression decision
        input_features = np.concatenate([
            self.embedding_matrix[node_idx],
            [compression_level / 10.0, memory_strength, utility_score],
            [node['metadata']['confidence']]
        ])

        # Label: optimal compression level
        access_frequency = self._calculate_access_frequency(node_idx)
        should_compress = (compression_level <= 5 and access_frequency < 0.3)

        return {
            'input': torch.FloatTensor(input_features),
            'task_type': 'compression',
            'compression_decision': torch.FloatTensor([1.0 if should_compress else 0.0]),
            'node_id': node['id']
        }

    def _create_temporal_sample(self, node_idx):
        """Create sample for temporal reasoning."""
        node = self.spatial_nodes[node_idx]
        last_accessed = datetime.datetime.fromisoformat(node['spatial']['last_accessed'])
        days_old = (datetime.datetime.now() - last_accessed).days

        # Features for temporal decision
        input_features = np.concatenate([
            self.embedding_matrix[node_idx],
            [days_old / 365.0],  # Normalized age
            [node['ai_native']['cognition']['memory_strength']]
        ])

        # Label: memorability score (combination of access recency and memory strength)
        memorability = (
            node['ai_native']['cognition']['memory_strength'] *
            (1.0 / (1.0 + days_old / 30.0))  # Exponential decay
        )

        return {
            'input': torch.FloatTensor(input_features),
            'task_type': 'temporal',
            'memorability_score': torch.FloatTensor([memorability]),
            'node_id': node['id']
        }

    def _calculate_access_frequency(self, node_idx):
        """Calculate access frequency for compression decisions."""
        # Simplified: based on relationships and memory strength
        node = self.spatial_nodes[node_idx]
        relationships = len(node.get('relations', {}).get('semantic', []))
        memory_strength = node['ai_native']['cognition']['memory_strength']

        return (relationships * 0.1 + memory_strength * 0.9) / 10.0


class SpatialReasoningModel(nn.Module):
    """Neural network for spatial reasoning and memory tasks."""

    def __init__(self, input_dim: int = 768 * 4 + 3, hidden_dim: int = 512):
        """Initialize spatial reasoning model."""
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Task-specific heads
        self.navigation_head = nn.Linear(hidden_dim // 2, 4)  # position + distance
        self.similarity_head = nn.Linear(hidden_dim // 2, 2)  # binary classification
        self.compression_head = nn.Linear(hidden_dim // 2, 1)  # sigmoid output
        self.temporal_head = nn.Linear(hidden_dim // 2, 1)    # memorability score

    def forward(self, x, task_type: str):
        """Forward pass for specified task type."""
        features = self.encoder(x)

        if task_type == 'navigation':
            output = self.navigation_head(features)
        elif task_type == 'similarity':
            output = self.similarity_head(features)
        elif task_type == 'compression':
            output = torch.sigmoid(self.compression_head(features))
        elif task_type == 'temporal':
            output = self.temporal_head(features)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return output


class SpatialTrainer:
    """Trainer for spatial memory and reasoning tasks."""

    def __init__(self, model, dataset, learning_rate: float = 1e-4):
        """Initialize trainer with model and dataset."""
        self.model = model
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Loss functions for different tasks
        self.navigation_loss = nn.MSELoss()
        self.similarity_loss = nn.CrossEntropyLoss()
        self.compression_loss = nn.BCELoss()
        self.temporal_loss = nn.MSELoss()

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        losses = {'total': 0.0, 'navigation': 0.0, 'similarity': 0.0, 'compression': 0.0, 'temporal': 0.0}
        counts = {'navigation': 0, 'similarity': 0, 'compression': 0, 'temporal': 0}

        # Use DataLoader for efficient batching
        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True, num_workers=0)

        for batch in dataloader:
            self.optimizer.zero_grad()

            inputs = batch['input'].to(self.device)
            task_type = batch['task_type'][0]  # All same in batch (we could optimize this)

            outputs = self.model(inputs, task_type)

            # Calculate loss based on task type
            if task_type == 'navigation':
                targets = torch.cat([batch['target_position'], batch['target_distance']], dim=-1).to(self.device)
                loss = self.navigation_loss(outputs, targets)
            elif task_type == 'similarity':
                targets = batch['similarity_label'].squeeze().to(self.device)
                loss = self.similarity_loss(outputs, targets)
            elif task_type == 'compression':
                targets = batch['compression_decision'].squeeze().to(self.device)
                loss = self.compression_loss(outputs, targets)
            elif task_type == 'temporal':
                targets = batch['memorability_score'].squeeze().to(self.device)
                loss = self.temporal_loss(outputs, targets)

            loss.backward()
            self.optimizer.step()

            losses[task_type] += loss.item()
            counts[task_type] += 1
            losses['total'] += loss.item()

        # Average losses
        for task in counts.keys():
            if counts[task] > 0:
                losses[task] /= counts[task]

        losses['total'] = losses['total'] / len(dataloader)
        return losses

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()
        results = {'accuracy': {}, 'loss': {}, 'samples': {}}

        with torch.no_grad():
            dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False)

            for task_type in ['navigation', 'similarity', 'compression', 'temporal']:
                task_samples = []
                task_predictions = []
                task_targets = []

                for batch in dataloader:
                    if batch['task_type'][0] != task_type:
                        continue

                    inputs = batch['input'].to(self.device)
                    outputs = self.model(inputs, task_type)

                    if task_type == 'navigation':
                        targets = torch.cat([batch['target_position'], batch['target_distance']], dim=-1)
                        predictions = outputs.cpu().numpy()
                        targets_np = targets.numpy()
                        accuracy = np.mean([
                            np.linalg.norm(pred[:3] - tgt[:3]) < 0.5 for pred, tgt in zip(predictions, targets_np)
                        ])
                    elif task_type == 'similarity':
                        targets = batch['similarity_label'].squeeze()
                        predictions = outputs.argmax(dim=1)
                        accuracy = (predictions == targets).float().mean().item()
                    elif task_type == 'compression':
                        targets = batch['compression_decision'].squeeze()
                        predictions = (outputs > 0.5).float().squeeze()
                        accuracy = (predictions == targets).float().mean().item()
                    elif task_type == 'temporal':
                        targets = batch['memorability_score'].squeeze()
                        predictions = outputs.squeeze()
                        accuracy = 1.0 - torch.mean(torch.abs(predictions - targets)).item()

                    if len(batch['input']) > 0:
                        task_samples.append(len(batch['input']))
                        task_predictions.extend(predictions.tolist() if hasattr(predictions, 'tolist') else [predictions])
                        task_targets.extend(targets.tolist() if hasattr(targets, 'tolist') else [targets.item()])

                if task_samples:
                    results['accuracy'][task_type] = accuracy
                    results['samples'][task_type] = sum(task_samples)

        return results

    def save_model(self, path: str):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")


def run_spatial_training(optimized_data_file: str = "data/optimized_knowledge_graph.json",
                        epochs: int = 50,
                        save_path: str = "data/spatial_memory_model.pth") -> Dict[str, Any]:
    """Run complete spatial memory training pipeline."""

    print("🧠 Starting Spatial Memory Training")
    print("=====================================")

    # Load optimized dataset from sleep-compute
    dataset = SpatialMemoryDataset(optimized_data_file)

    if len(dataset) == 0:
        print("❌ No valid data for training")
        return {}

    # Initialize model and trainer
    input_dim = len(dataset[0]['input'])
    model = SpatialReasoningModel(input_dim=input_dim)
    trainer = SpatialTrainer(model, dataset)

    print(f"📊 Training on {len(dataset)} samples")
    print(f"🧠 Model input dim: {input_dim}")

    # Training loop
    training_history = []

    for epoch in range(epochs):
        losses = trainer.train_epoch()

        if epoch % 10 == 0:
            eval_results = trainer.evaluate()
            accuracy_summary = {k: v for k, v in eval_results['accuracy'].items()}

            print(f"Epoch {epoch:3d}: Loss={losses['total']:.4f} | "
                  f"Acc={accuracy_summary if accuracy_summary else 'N/A'}")

            training_history.append({
                'epoch': epoch,
                'losses': losses,
                'evaluation': eval_results
            })

    # Final evaluation
    print("\n🔬 Final Evaluation")
    final_results = trainer.evaluate()
    print("Spatial Navigation Accuracy:", final_results['accuracy'].get('navigation', 'N/A'))
    print("Semantic Similarity Accuracy:", final_results['accuracy'].get('similarity', 'N/A'))
    print("Compression Decision Accuracy:", final_results['accuracy'].get('compression', 'N/A'))
    print("Temporal Memory Accuracy:", final_results['accuracy'].get('temporal', 'N/A'))

    # Save model
    trainer.save_model(save_path)

    results_summary = {
        'training_history': training_history,
        'final_evaluation': final_results,
        'model_path': save_path,
        'dataset_size': len(dataset),
        'training_time': datetime.datetime.now().isoformat()
    }

    # Save training summary
    with open("data/spatial_training_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"\n💾 Results saved to data/spatial_training_results.json")
    print("🎉 Spatial Memory Training Complete!")

    return results_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="K3D Spatial Memory Trainer")
    parser.add_argument("--data", default="data/optimized_knowledge_graph.json",
                       help="Path to optimized knowledge graph")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--model-path", default="data/spatial_memory_model.pth",
                       help="Path to save trained model")

    args = parser.parse_args()

    results = run_spatial_training(args.data, args.epochs, args.model_path)
