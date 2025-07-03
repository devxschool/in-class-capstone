import asyncio
from typing import List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import time

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from app.core.models import DocumentChunk, EmbeddingResult
from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles embedding generation using OpenAI's embedding model."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        self.batch_size = 100  # OpenAI's recommended batch size
    
    def _prepare_documents(self, chunks: List[DocumentChunk]) -> List[Document]:
        """Convert DocumentChunk objects to LangChain Document objects."""
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.id,
                    "source_file": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    **chunk.metadata
                }
            )
            documents.append(doc)
        return documents
    
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[EmbeddingResult]:
        """Generate embeddings for a list of document chunks."""
        if not chunks:
            return []
        
        try:
            documents = self._prepare_documents(chunks)
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                batch_embeddings = self.embeddings.embed_documents(
                    [doc.page_content for doc in batch]
                )
                all_embeddings.extend(batch_embeddings)
            
            # Create EmbeddingResult objects
            results = []
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                result = EmbeddingResult(
                    chunk_id=chunk.id,
                    embedding=embedding,
                    metadata={
                        "source_file": chunk.source_file,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        **chunk.metadata
                    }
                )
                results.append(result)
            
            logger.info(f"Successfully generated embeddings for {len(results)} chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def generate_embeddings_async(self, chunks: List[DocumentChunk]) -> List[EmbeddingResult]:
        """Generate embeddings asynchronously."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, self.generate_embeddings, chunks
            ) 