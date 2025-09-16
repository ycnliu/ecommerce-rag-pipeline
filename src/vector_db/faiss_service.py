"""
FAISS vector database service for similarity search.
"""
import os
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import faiss
from loguru import logger

from ..utils.exceptions import VectorDBError
from ..data.models import ProductMetadata, IndexStats


class FAISSVectorDB:
    """FAISS-based vector database for similarity search."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        metric: str = "l2",
        nlist: int = 100,
        nprobe: int = 10
    ):
        """
        Initialize FAISS vector database.

        Args:
            dimension: Embedding dimension
            index_type: Type of index ("flat", "ivf", "hnsw")
            metric: Distance metric ("l2" or "ip" for inner product)
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search for IVF index
        """
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.metric = metric.lower()
        self.nlist = nlist
        self.nprobe = nprobe

        self.index = None
        self.metadata: List[ProductMetadata] = []
        self.is_trained = False

        self._create_index()

    def _create_index(self) -> None:
        """Create FAISS index based on specified type."""
        try:
            if self.index_type == "flat":
                if self.metric == "l2":
                    self.index = faiss.IndexFlatL2(self.dimension)
                elif self.metric == "ip":
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    raise ValueError(f"Unsupported metric for flat index: {self.metric}")

            elif self.index_type == "ivf":
                # IVF (Inverted File) index for faster approximate search
                quantizer = faiss.IndexFlatL2(self.dimension)
                if self.metric == "l2":
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
                elif self.metric == "ip":
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
                else:
                    raise ValueError(f"Unsupported metric for IVF index: {self.metric}")

                self.index.nprobe = self.nprobe

            elif self.index_type == "hnsw":
                # Hierarchical Navigable Small World for very fast approximate search
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = 40
                self.index.hnsw.efSearch = 16

            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")

            logger.info(f"Created FAISS {self.index_type} index with dimension {self.dimension}")

        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise VectorDBError(f"Failed to create FAISS index: {e}") from e

    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata: List[ProductMetadata],
        train_if_needed: bool = True
    ) -> None:
        """
        Add vectors and metadata to the index.

        Args:
            embeddings: Array of embeddings with shape (n, dimension)
            metadata: List of metadata objects corresponding to embeddings
            train_if_needed: Whether to train the index if needed
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")

        try:
            # Ensure embeddings are float32
            embeddings = embeddings.astype('float32')

            # Train index if needed (for IVF)
            if self.index_type == "ivf" and not self.is_trained:
                if train_if_needed and len(embeddings) >= self.nlist:
                    logger.info("Training IVF index...")
                    self.index.train(embeddings)
                    self.is_trained = True
                elif not train_if_needed:
                    raise VectorDBError("IVF index needs training but train_if_needed=False")
                else:
                    raise VectorDBError(f"Need at least {self.nlist} vectors to train IVF index")

            # Add vectors to index
            self.index.add(embeddings)
            self.metadata.extend(metadata)

            logger.info(f"Added {len(embeddings)} vectors to index. Total: {self.index.ntotal}")

        except Exception as e:
            logger.error(f"Failed to add vectors to index: {e}")
            raise VectorDBError(f"Failed to add vectors to index: {e}") from e

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        return_distances: bool = True
    ) -> Tuple[List[ProductMetadata], Optional[List[float]]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query embedding with shape (1, dimension) or (dimension,)
            k: Number of nearest neighbors to return
            return_distances: Whether to return distances

        Returns:
            Tuple of (metadata_list, distances) or (metadata_list, None)
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return [], None if return_distances else []

        try:
            # Ensure query is 2D and float32
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding.astype('float32')

            # Perform search
            distances, indices = self.index.search(query_embedding, k)

            # Get metadata for found indices
            result_metadata = []
            result_distances = []

            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.metadata):  # Valid index
                    result_metadata.append(self.metadata[idx])
                    if return_distances:
                        result_distances.append(float(distances[0][i]))

            return result_metadata, result_distances if return_distances else None

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorDBError(f"Search failed: {e}") from e

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> List[Tuple[List[ProductMetadata], List[float]]]:
        """
        Perform batch search for multiple queries.

        Args:
            query_embeddings: Query embeddings with shape (n_queries, dimension)
            k: Number of nearest neighbors per query

        Returns:
            List of (metadata_list, distances) for each query
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return [([], []) for _ in range(len(query_embeddings))]

        try:
            query_embeddings = query_embeddings.astype('float32')
            distances, indices = self.index.search(query_embeddings, k)

            results = []
            for i in range(len(query_embeddings)):
                result_metadata = []
                result_distances = []

                for j, idx in enumerate(indices[i]):
                    if idx >= 0 and idx < len(self.metadata):
                        result_metadata.append(self.metadata[idx])
                        result_distances.append(float(distances[i][j]))

                results.append((result_metadata, result_distances))

            return results

        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            raise VectorDBError(f"Batch search failed: {e}") from e

    def save_index(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save index to disk.

        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata (optional)
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            logger.info(f"Saved FAISS index to {index_path}")

            # Save metadata if path provided
            if metadata_path:
                import pickle
                with open(metadata_path, 'wb') as f:
                    pickle.dump(self.metadata, f)
                logger.info(f"Saved metadata to {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise VectorDBError(f"Failed to save index: {e}") from e

    def load_index(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Load index from disk.

        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file (optional)
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            self.is_trained = True  # Assume loaded index is trained
            logger.info(f"Loaded FAISS index from {index_path}")

            # Load metadata if path provided
            if metadata_path and os.path.exists(metadata_path):
                import pickle
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded metadata from {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise VectorDBError(f"Failed to load index: {e}") from e

    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        memory_usage = None
        if hasattr(self.index, 'sa_code_size'):  # For some index types
            try:
                # Rough estimation of memory usage
                memory_usage = (self.index.ntotal * self.dimension * 4) / (1024 * 1024)  # MB
            except:
                pass

        return IndexStats(
            total_vectors=self.index.ntotal,
            dimension=self.dimension,
            index_type=self.index_type,
            memory_usage_mb=memory_usage
        )

    def remove_vectors(self, indices: List[int]) -> None:
        """
        Remove vectors by indices.

        Note: This creates a new index without the specified vectors.
        Only supported for flat indices in this implementation.
        """
        if self.index_type != "flat":
            raise VectorDBError("Vector removal only supported for flat indices")

        try:
            # Get all vectors except the ones to remove
            all_vectors = []
            new_metadata = []

            for i in range(self.index.ntotal):
                if i not in indices:
                    # Reconstruct vector (only works for flat indices)
                    vector = self.index.reconstruct(i)
                    all_vectors.append(vector)
                    new_metadata.append(self.metadata[i])

            # Create new index with remaining vectors
            if all_vectors:
                embeddings = np.vstack(all_vectors)
                self._create_index()  # Reset index
                self.add_vectors(embeddings, new_metadata, train_if_needed=True)
            else:
                self._create_index()  # Empty index
                self.metadata = []

            logger.info(f"Removed {len(indices)} vectors from index")

        except Exception as e:
            logger.error(f"Failed to remove vectors: {e}")
            raise VectorDBError(f"Failed to remove vectors: {e}") from e

    def clear(self) -> None:
        """Clear the index and metadata."""
        self._create_index()
        self.metadata = []
        logger.info("Cleared index and metadata")