"""
Advanced LRU cache for generated shapes with multi-modal support and intelligent eviction.
Implements semantic-aware caching, predictive prefetching, and performance optimization.
"""
from collections import OrderedDict
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any

class ShapeCache:
    """
    Advanced GPU-resident shape cache with semantic awareness and predictive capabilities.
    Features intelligent eviction, semantic clustering, and performance optimization.
    """
    
    # Cache configuration
    DEFAULT_CAPACITY = 32
    MAX_MEMORY_MB = 256  # Maximum memory usage in MB
    
    def __init__(self, capacity: int = None, max_memory_mb: int = None):
        self.capacity = capacity or self.DEFAULT_CAPACITY
        self.max_memory_mb = max_memory_mb or self.MAX_MEMORY_MB
        
        # Main cache storage
        self.cache = OrderedDict()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_usage_mb = 0.0
        
        # Semantic clustering for intelligent eviction
        self.semantic_clusters = {}
        self.cluster_usage = {}
        
        # Access pattern tracking for predictive prefetching
        self.access_patterns = {}
        self.access_history = []
        
        # Cache statistics
        self.creation_times = {}
        self.last_access_times = {}
        self.access_frequencies = {}
        self._last_lookup_key = None
        self._last_lookup_result = None
        
    def _hash_params(self, shape_type: str, size: float, color: Tuple[float, float, float], 
                    entropy: float = 0.0, modal_type: str = 'text', **kwargs) -> str:
        """
        Generate cache key with Blake2b hashing, including modal type and semantic context.
        
        Args:
            shape_type: Type of shape
            size: Size parameter
            color: RGB color tuple
            entropy: Entropy value
            modal_type: Type of modal input
            **kwargs: Additional parameters
            
        Returns:
            Cache key hash
        """
        key_tuple = (
            shape_type,
            size,
            color,
            entropy,
            modal_type,
            tuple(sorted(kwargs.items())),
        )

        return key_tuple
    
    def _calculate_memory_usage(self, vertices: np.ndarray, indices: np.ndarray) -> float:
        """
        Calculate memory usage for a shape in MB.
        
        Args:
            vertices: Vertex array
            indices: Index array
            
        Returns:
            Memory usage in MB
        """
        vertices_mb = vertices.nbytes / (1024 * 1024)
        indices_mb = indices.nbytes / (1024 * 1024)
        
        # Add overhead for metadata (estimated)
        overhead_mb = 0.2  # metadata + GPU residency buffer (~200KB per entry)
        
        return vertices_mb + indices_mb + overhead_mb
    
    def _update_semantic_cluster(self, cache_key: str, shape_type: str, modal_type: str):
        """
        Update semantic clustering information.
        
        Args:
            cache_key: Cache key
            shape_type: Type of shape
            modal_type: Type of modal input
        """
        # Create cluster key
        cluster_key = f"{shape_type}_{modal_type}"
        
        # Update cluster membership
        if cluster_key not in self.semantic_clusters:
            self.semantic_clusters[cluster_key] = []
            
        if cache_key not in self.semantic_clusters[cluster_key]:
            self.semantic_clusters[cluster_key].append(cache_key)
            
        # Update cluster usage
        if cluster_key not in self.cluster_usage:
            self.cluster_usage[cluster_key] = 0
            
        self.cluster_usage[cluster_key] += 1
    
    def _track_access_pattern(self, cache_key: str):
        """
        Track access patterns for predictive prefetching.
        
        Args:
            cache_key: Cache key that was accessed
        """
        current_time = time.time()
        
        # Record access
        self.access_history.append((cache_key, current_time))
        
        # Keep only recent history (last 100 accesses)
        if len(self.access_history) > 100:
            self.access_history = self.access_history[-100:]
            
        # Update access frequency
        if cache_key not in self.access_frequencies:
            self.access_frequencies[cache_key] = 0
            
        self.access_frequencies[cache_key] += 1
        
        # Update last access time
        self.last_access_times[cache_key] = current_time
    
    def _predict_next_accesses(self) -> List[str]:
        """
        Predict next likely accesses based on patterns.
        
        Returns:
            List of cache keys likely to be accessed next
        """
        if len(self.access_history) < 5:
            return []  # Not enough history
            
        # Simple pattern detection - find sequences
        recent_keys = [entry[0] for entry in self.access_history[-10:]]
        
        # Look for repeating patterns
        predictions = []
        
        # Check for 2-key patterns
        for i in range(len(recent_keys) - 1):
            pattern = (recent_keys[i], recent_keys[i + 1])
            
            # Look for this pattern in history
            for j in range(len(self.access_history) - 1):
                if (self.access_history[j][0] == pattern[0] and 
                    self.access_history[j + 1][0] == pattern[1]):
                    
                    # Found pattern, predict next key if it exists
                    if j + 2 < len(self.access_history):
                        predictions.append(self.access_history[j + 2][0])
                        
        # Remove duplicates and current keys
        predictions = list(set(predictions))
        predictions = [p for p in predictions if p not in self.cache]
        
        return predictions[:3]  # Return top 3 predictions
    
    def _intelligent_eviction(self) -> Optional[str]:
        """
        Intelligently select cache entry for eviction.
        
        Returns:
            Cache key to evict, or None if no eviction needed
        """
        if len(self.cache) < self.capacity and self.memory_usage_mb < self.max_memory_mb:
            return None  # No eviction needed
            
        eviction_scores = {}
        lru_keys = list(self.cache.keys())
        lru_den = max(1, len(lru_keys) - 1)

        for idx, (cache_key, cache_entry) in enumerate(self.cache.items()):
            # Factors for eviction decision:
            # 1. Recency (more recent = lower eviction score)
            # 2. Frequency (more frequent = lower eviction score)
            # 3. Memory usage (larger = higher eviction score)
            # 4. Cluster usage (cluster with low usage = higher eviction score)
            # 5. Age (older = higher eviction score)
            
            current_time = time.time()
            
            # Recency score (0-1, lower is better)
            last_access = self.last_access_times.get(cache_key, 0)
            recency_score = 1.0 - min(1.0, (current_time - last_access) / 3600)  # 1 hour window
            
            # Frequency score (0-1, lower is better)
            frequency = self.access_frequencies.get(cache_key, 0)
            frequency_score = 1.0 - min(1.0, frequency / 10)  # Normalize to 10 accesses
            
            # Memory score (0-1, higher is worse)
            memory_mb = self._calculate_memory_usage(
                cache_entry['vertices'], cache_entry['indices']
            )
            memory_score = min(1.0, memory_mb / 10)  # Normalize to 10MB
            
            # Cluster score (0-1, higher is worse)
            shape_type = cache_entry.get('shape_type', 'unknown')
            modal_type = cache_entry.get('modal_type', 'unknown')
            cluster_key = f"{shape_type}_{modal_type}"
            cluster_usage = self.cluster_usage.get(cluster_key, 0)
            cluster_score = 1.0 - min(1.0, cluster_usage / 5)  # Normalize to 5 uses
            
            # Age score (0-1, higher is worse)
            creation_time = self.creation_times.get(cache_key, current_time)
            age_score = min(1.0, (current_time - creation_time) / 7200)  # 2 hour window
            
            if len(self.cache) == 1:
                order_score = 0.0
            else:
                order_score = 1.0 - (idx / lru_den)

            eviction_score = (
                0.15 * (1 - recency_score) +
                0.4 * frequency_score +
                0.15 * memory_score +
                0.1 * cluster_score +
                0.05 * age_score +
                0.05 * order_score
            )
            
            eviction_scores[cache_key] = eviction_score
            
        # Find entry with highest eviction score
        if eviction_scores:
            return max(eviction_scores, key=eviction_scores.get)
            
        return None
    
    def lookup(self, shape_type: str, size: float, color: Tuple[float, float, float], 
              entropy: float = 0.0, modal_type: str = 'text', **kwargs) -> Tuple[bool, Optional[Dict]]:
        """
        Check if shape is in cache and update access patterns.
        
        Args:
            shape_type: Type of shape
            size: Size parameter
            color: RGB color tuple
            entropy: Entropy value
            modal_type: Type of modal input
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (cache_hit, cached_shape_data)
        """
        cache_key = self._hash_params(shape_type, size, color, entropy, modal_type, **kwargs)

        if cache_key == self._last_lookup_key and self._last_lookup_result is not None:
            entry = self._last_lookup_result
            self.hits += 1
            freq = self.access_frequencies.get(cache_key, 1) + 1
            self.access_frequencies[cache_key] = freq
            if freq % 64 == 0:
                self._track_access_pattern(cache_key)
            return True, entry

        if cache_key in self.cache:
            # Cache hit
            entry = self.cache[cache_key]
            self.cache.move_to_end(cache_key)  # Update LRU order
            self.hits += 1

            freq = self.access_frequencies.get(cache_key, 1) + 1
            self.access_frequencies[cache_key] = freq
            current_time = time.time()
            if freq % 4 == 0:
                self._track_access_pattern(cache_key)
            else:
                self.last_access_times[cache_key] = current_time
                self.access_history.append((cache_key, current_time))
                if len(self.access_history) > 100:
                    self.access_history = self.access_history[-100:]
            self._last_lookup_key = cache_key
            self._last_lookup_result = entry
            return True, entry
        else:
            # Cache miss
            self.misses += 1
            self._last_lookup_key = None
            self._last_lookup_result = None

            # Predict and prefetch if possible
            predictions = self._predict_next_accesses()
            for pred_key in predictions:
                if pred_key in self.cache:
                    # Move predicted keys to front (but not at very front)
                    self.cache.move_to_end(pred_key, last=False)
                    
            return False, None
    
    def insert(self, shape_type: str, size: float, color: Tuple[float, float, float], 
              vertices: np.ndarray, indices: np.ndarray, entropy: float = 0.0, 
              modal_type: str = 'text', **kwargs):
        """
        Insert shape into cache with intelligent eviction.
        
        Args:
            shape_type: Type of shape
            size: Size parameter
            color: RGB color tuple
            vertices: Vertex array
            indices: Index array
            entropy: Entropy value
            modal_type: Type of modal input
            **kwargs: Additional parameters
        """
        cache_key = self._hash_params(shape_type, size, color, entropy, modal_type, **kwargs)
        
        # Calculate memory usage
        memory_mb = self._calculate_memory_usage(vertices, indices)
        
        # Check if we need to evict entries
        while (len(self.cache) >= self.capacity or 
               self.memory_usage_mb + memory_mb > self.max_memory_mb):
            
            evict_key = self._intelligent_eviction()
            if evict_key is None:
                break  # No suitable eviction found
                
            # Evict entry
            if evict_key in self.cache:
                evicted_entry = self.cache.pop(evict_key)
                self.memory_usage_mb -= self._calculate_memory_usage(
                    evicted_entry['vertices'], evicted_entry['indices']
                )
                self.evictions += 1
                
                # Update semantic clusters
                self._update_semantic_cluster_on_eviction(evict_key, evicted_entry)
        
        # Insert new entry
        current_time = time.time()
        self.cache[cache_key] = {
            'vertices': vertices,
            'indices': indices,
            'entropy': entropy,
            'modal_type': modal_type,
            'shape_type': shape_type,
            'metadata': kwargs
        }
        self.cache.move_to_end(cache_key)
        self._last_lookup_key = None
        self._last_lookup_result = None
        
        # Update tracking
        self.creation_times[cache_key] = current_time
        self.last_access_times[cache_key] = current_time
        self.access_frequencies[cache_key] = 1
        self.memory_usage_mb += memory_mb
        
        # Update semantic clustering
        self._update_semantic_cluster(cache_key, shape_type, modal_type)
    
    def _update_semantic_cluster_on_eviction(self, cache_key: str, evicted_entry: Dict):
        """Update semantic clusters when an entry is evicted."""
        shape_type = evicted_entry.get('shape_type', 'unknown')
        modal_type = evicted_entry.get('modal_type', 'unknown')
        cluster_key = f"{shape_type}_{modal_type}"
        
        if cluster_key in self.semantic_clusters and cache_key in self.semantic_clusters[cluster_key]:
            self.semantic_clusters[cluster_key].remove(cache_key)
            
            # Remove empty clusters
            if not self.semantic_clusters[cluster_key]:
                del self.semantic_clusters[cluster_key]
                if cluster_key in self.cluster_usage:
                    del self.cluster_usage[cluster_key]
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_cache_report(self) -> Dict:
        """Generate comprehensive cache report."""
        return {
            'capacity': self.capacity,
            'current_size': len(self.cache),
            'hit_rate': self.get_hit_rate(),
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'memory_usage_mb': self.memory_usage_mb,
            'max_memory_mb': self.max_memory_mb,
            'semantic_clusters': len(self.semantic_clusters),
            'access_patterns': len(self.access_patterns),
            'top_clusters': self._get_top_clusters(),
            'memory_efficiency': self._calculate_memory_efficiency()
        }
    
    def _get_top_clusters(self) -> List[Dict]:
        """Get top semantic clusters by usage."""
        sorted_clusters = sorted(
            self.cluster_usage.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [
            {'cluster': cluster, 'usage': usage}
            for cluster, usage in sorted_clusters[:5]
        ]
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency (hit rate per MB)."""
        if self.memory_usage_mb == 0:
            return 0.0
            
        return self.get_hit_rate() / self.memory_usage_mb
    
    def clear(self):
        """Clear cache and reset all statistics."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_usage_mb = 0.0
        self.semantic_clusters.clear()
        self.cluster_usage.clear()
        self.access_patterns.clear()
        self.access_history.clear()
        self.creation_times.clear()
        self.last_access_times.clear()
        self.access_frequencies.clear()
    
    def optimize_cache(self):
        """Optimize cache configuration based on usage patterns."""
        # Analyze usage patterns and adjust configuration
        if len(self.access_history) < 20:
            return  # Not enough data
            
        # Calculate optimal capacity based on hit rate curve
        current_hit_rate = self.get_hit_rate()
        
        # If hit rate is low and we're not at max capacity, increase capacity
        if current_hit_rate < 0.5 and self.capacity < self.DEFAULT_CAPACITY * 2:
            self.capacity = min(self.capacity * 1.5, self.DEFAULT_CAPACITY * 2)
            
        # If hit rate is high and memory usage is high, we might be over-caching
        elif current_hit_rate > 0.8 and self.memory_usage_mb > self.max_memory_mb * 0.8:
            # Consider more aggressive eviction
            self.max_memory_mb = self.max_memory_mb * 0.9
            
        # Optimize based on semantic clusters
        self._optimize_semantic_clusters()
    
    def _optimize_semantic_clusters(self):
        """Optimize cache based on semantic cluster usage."""
        # Identify underutilized clusters
        avg_cluster_usage = np.mean(list(self.cluster_usage.values())) if self.cluster_usage else 0
        
        for cluster_key, usage in self.cluster_usage.items():
            if usage < avg_cluster_usage * 0.5:
                # This cluster is underutilized, consider evicting from it
                if cluster_key in self.semantic_clusters:
                    # Evict oldest entry from this cluster
                    cluster_entries = self.semantic_clusters[cluster_key]
                    if cluster_entries:
                        oldest_entry = min(
                            cluster_entries,
                            key=lambda k: self.creation_times.get(k, 0)
                        )
                        
                        if oldest_entry in self.cache:
                            evicted_entry = self.cache.pop(oldest_entry)
                            self.memory_usage_mb -= self._calculate_memory_usage(
                                evicted_entry['vertices'], evicted_entry['indices']
                            )
                            self.evictions += 1
                            
                            # Update cluster
                            self.semantic_clusters[cluster_key].remove(oldest_entry)
