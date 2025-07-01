import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.workflow import DocumentProcessingWorkflow
from app.core.models import ProcessingResult
from app.vectorstore.pinecone_service import PineconeService
from config import settings

logger = logging.getLogger(__name__)


class IngestService:
    """Main service for ingesting PDF documents into the vector database."""
    
    def __init__(self):
        self.workflow = DocumentProcessingWorkflow()
        self.pinecone_service = PineconeService()
    
    def process_single_document(self, file_path: Path) -> ProcessingResult:
        """Process a single PDF document."""
        try:
            logger.info(f"Starting ingestion for: {file_path}")
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not file_path.suffix.lower() == '.pdf':
                raise ValueError(f"File must be a PDF: {file_path}")
            
            # Process the document through the workflow
            result = self.workflow.process_document(file_path)
            
            if result.success:
                logger.info(f"Successfully processed {file_path.name}: {result.chunks_created} chunks created")
            else:
                logger.error(f"Failed to process {file_path.name}: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {str(e)}")
            return ProcessingResult(
                success=False,
                document_id="unknown",
                chunks_created=0,
                error_message=str(e),
                processing_time=0.0,
                metadata=None
            )
    
    def process_multiple_documents(self, file_paths: List[Path]) -> List[ProcessingResult]:
        """Process multiple PDF documents sequentially."""
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_single_document(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results.append(ProcessingResult(
                    success=False,
                    document_id="unknown",
                    chunks_created=0,
                    error_message=str(e),
                    processing_time=0.0,
                    metadata=None
                ))
        
        return results
    
    async def process_documents_async(self, file_paths: List[Path], max_workers: int = 3) -> List[ProcessingResult]:
        """Process multiple PDF documents asynchronously with limited concurrency."""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, self.process_single_document, file_path)
                for file_path in file_paths
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions that occurred
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Exception processing {file_paths[i]}: {str(result)}")
                    processed_results.append(ProcessingResult(
                        success=False,
                        document_id="unknown",
                        chunks_created=0,
                        error_message=str(result),
                        processing_time=0.0,
                        metadata=None
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
    
    def process_directory(self, directory_path: Path, recursive: bool = False) -> List[ProcessingResult]:
        """Process all PDF files in a directory."""
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Find all PDF files
        if recursive:
            pdf_files = list(directory_path.rglob("*.pdf"))
        else:
            pdf_files = list(directory_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        return self.process_multiple_documents(pdf_files)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        try:
            stats = self.pinecone_service.get_index_stats()
            return {
                "vector_database_stats": stats,
                "embedding_model": settings.embedding_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap
            }
        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            return {"error": str(e)}
    
    def search_documents(self, query: str, top_k: int = 5, 
                        filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for documents using a text query."""
        try:
            # Generate embedding for the query
            from app.core.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            
            # Create a dummy chunk for the query
            from app.core.models import DocumentChunk
            query_chunk = DocumentChunk(
                content=query,
                chunk_index=0,
                source_file="query"
            )
            
            # Generate embedding
            query_embedding_result = embedding_service.generate_embeddings([query_chunk])[0]
            
            # Search in Pinecone
            results = self.pinecone_service.search_similar(
                query_embedding_result.embedding,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def delete_document(self, file_hash: str) -> Dict[str, Any]:
        """Delete all vectors associated with a specific document."""
        try:
            result = self.pinecone_service.delete_by_file_hash(file_hash)
            logger.info(f"Deleted document with hash: {file_hash}")
            return result
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return {"error": str(e)}
