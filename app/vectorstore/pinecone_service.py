import logging
from typing import List, Dict, Any, Optional
import time

from pinecone import Pinecone, ServerlessSpec

from app.core.models import EmbeddingResult, DocumentChunk
from config import settings

logger = logging.getLogger(__name__)


class PineconeService:
    """Handles Pinecone vector database operations."""
    
    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self._ensure_index_exists()
    
    def _ensure_index_exists(self) -> None:
        """Ensure the Pinecone index exists, create if it doesn't."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                
                # Create index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=3072,  # text-embedding-3-large dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status["ready"]:
                    time.sleep(1)
                    logger.info("Waiting for index to be ready...")
                
                logger.info(f"Successfully created index: {self.index_name}")
            else:
                logger.info(f"Using existing index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring index exists: {str(e)}")
            raise
    
    def _prepare_vectors_for_upsert(self, embeddings: List[EmbeddingResult]) -> List[Dict[str, Any]]:
        """Prepare embedding results for Pinecone upsert operation."""
        vectors = []
        for embedding_result in embeddings:
            vector_data = {
                "id": embedding_result.chunk_id,
                "values": embedding_result.embedding,
                "metadata": {
                    "content": embedding_result.metadata.get("content", ""),  # Store actual content
                    "source_file": embedding_result.metadata.get("source_file", ""),
                    "chunk_index": embedding_result.metadata.get("chunk_index", 0),
                    "filename": embedding_result.metadata.get("filename", ""),
                    "file_hash": embedding_result.metadata.get("file_hash", ""),
                    "total_chunks": embedding_result.metadata.get("total_chunks", 0)
                }
            }
            vectors.append(vector_data)
        return vectors
    
    def upsert_embeddings(self, embeddings: List[EmbeddingResult]) -> Dict[str, Any]:
        """Upsert embeddings to Pinecone index."""
        if not embeddings:
            return {"upserted_count": 0}
        
        try:
            index = self.pc.Index(self.index_name)
            vectors = self._prepare_vectors_for_upsert(embeddings)
            
            logger.info(f"Upserting {len(vectors)} vectors to Pinecone")
            
            # Upsert in batches (Pinecone recommends batches of 100)
            batch_size = 100
            total_upserted = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                result = index.upsert(vectors=batch)
                total_upserted += result.upserted_count
                logger.info(f"Upserted batch {i//batch_size + 1}: {result.upserted_count} vectors")
            
            logger.info(f"Successfully upserted {total_upserted} vectors to Pinecone")
            return {"upserted_count": total_upserted}
            
        except Exception as e:
            logger.error(f"Error upserting embeddings to Pinecone: {str(e)}")
            raise
    
    def delete_by_file_hash(self, file_hash: str) -> Dict[str, Any]:
        """Delete all vectors associated with a specific file hash."""
        try:
            index = self.pc.Index(self.index_name)
            
            # First check if any vectors exist with this file hash
            try:
                # Try to query for vectors with this file hash to see if any exist
                dummy_vector = 3072
                query_result = index.query(
                    vector=dummy_vector,
                    top_k=1,
                    filter={"file_hash": {"$eq": file_hash}},
                    include_metadata=False
                )
                
                # If no vectors found, return early
                if not query_result.matches:
                    logger.info(f"No vectors found for file hash {file_hash}, skipping delete")
                    return {"deleted_count": 0}
                
            except Exception as query_error:
                # If query fails (e.g., no vectors exist), just log and continue
                logger.info(f"Query check failed for file hash {file_hash}, assuming no vectors exist: {str(query_error)}")
                return {"deleted_count": 0}
            
            # Delete vectors by metadata filter
            result = index.delete(
                filter={"file_hash": {"$eq": file_hash}}
            )
            
            logger.info(f"Deleted vectors for file hash {file_hash}")
            return {"deleted_count": result.deleted_count}
            
        except Exception as e:
            logger.error(f"Error deleting vectors by file hash: {str(e)}")
            # Don't raise the error, just return empty result
            return {"deleted_count": 0}
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, 
                      filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in the index."""
        try:
            index = self.pc.Index(self.index_name)
            
            search_kwargs = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True
            }
            
            if filter_dict:
                search_kwargs["filter"] = filter_dict
            
            results = index.query(**search_kwargs)
            
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
            
        except Exception as e:
            logger.error(f"Error searching Pinecone index: {str(e)}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            raise 