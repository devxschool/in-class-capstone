import time
import logging
import traceback
from typing import Dict, Any, List, TypedDict
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.core.models import DocumentChunk, DocumentMetadata, ProcessingResult, EmbeddingResult
from app.core.pdf_parser import PDFParser
from app.core.embedding_service import EmbeddingService
from app.vectorstore.pinecone_service import PineconeService

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State dictionary for the workflow."""
    file_path: str
    chunks: List[DocumentChunk]
    metadata: DocumentMetadata
    embeddings: List[EmbeddingResult]
    upsert_result: Dict[str, Any]
    error: str
    processing_time: float
    start_time: float


class DocumentProcessingWorkflow:
    """LangGraph workflow for processing PDF documents."""
    
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.embedding_service = EmbeddingService()
        self.pinecone_service = PineconeService()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("parse_pdf", self._parse_pdf_node)
        workflow.add_node("generate_embeddings", self._generate_embeddings_node)
        workflow.add_node("upsert_to_pinecone", self._upsert_to_pinecone_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("parse_pdf")
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "parse_pdf",
            self._should_continue,
            {
                "continue": "generate_embeddings",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_embeddings",
            self._should_continue,
            {
                "continue": "upsert_to_pinecone",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "upsert_to_pinecone",
            self._should_continue,
            {
                "continue": END,
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _parse_pdf_node(self, state: WorkflowState) -> WorkflowState:
        """Parse PDF and extract chunks."""
        try:
            file_path = Path(state["file_path"])
            logger.info(f"Starting PDF parsing for: {file_path}")
            
            chunks, metadata = self.pdf_parser.parse_pdf(file_path)
            
            state["chunks"] = chunks
            state["metadata"] = metadata
            if metadata:
                metadata.processed_at = time.time()
            
            logger.info(f"Successfully parsed PDF: {len(chunks)} chunks created")
            
        except Exception as e:
            logger.error(f"Error in PDF parsing: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            state["error"] = str(e)
        
        return state
    
    def _generate_embeddings_node(self, state: WorkflowState) -> WorkflowState:
        """Generate embeddings for document chunks."""
        try:
            chunks = state.get("chunks", [])
            if not chunks:
                raise ValueError("No chunks available for embedding generation")
            
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            embeddings = self.embedding_service.generate_embeddings(chunks)
            state["embeddings"] = embeddings
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            
        except Exception as e:
            logger.error(f"Error in embedding generation: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            state["error"] = str(e)
        
        return state
    
    def _upsert_to_pinecone_node(self, state: WorkflowState) -> WorkflowState:
        """Upsert embeddings to Pinecone."""
        try:
            embeddings = state.get("embeddings", [])
            if not embeddings:
                raise ValueError("No embeddings available for upsert")
            
            logger.info(f"Upserting {len(embeddings)} embeddings to Pinecone")
            
            # Delete existing vectors for this file (if any) - handle gracefully
            metadata = state.get("metadata")
            if metadata and metadata.file_hash:
                try:
                    delete_result = self.pinecone_service.delete_by_file_hash(metadata.file_hash)
                    logger.info(f"Delete operation result: {delete_result}")
                except Exception as delete_error:
                    logger.warning(f"Failed to delete existing vectors, continuing with upsert: {str(delete_error)}")
            
            # Upsert new embeddings
            upsert_result = self.pinecone_service.upsert_embeddings(embeddings)
            state["upsert_result"] = upsert_result
            
            logger.info(f"Successfully upserted embeddings to Pinecone")
            
        except Exception as e:
            logger.error(f"Error in Pinecone upsert: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            state["error"] = str(e)
        
        return state
    
    def _handle_error_node(self, state: WorkflowState) -> WorkflowState:
        """Handle errors in the workflow."""
        error = state.get("error", "Unknown error")
        logger.error(f"Workflow error: {error}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return state
    
    def _should_continue(self, state: WorkflowState) -> str:
        """Determine if the workflow should continue or handle error."""
        return "error" if state.get("error") else "continue"
    
    def process_document(self, file_path: Path) -> ProcessingResult:
        """Process a PDF document through the complete pipeline."""
        start_time = time.time()
        
        try:
            # Initialize state as dictionary
            initial_state: WorkflowState = {
                "file_path": str(file_path),
                "chunks": [],
                "metadata": None,
                "embeddings": [],
                "upsert_result": {},
                "error": None,
                "processing_time": 0.0,
                "start_time": start_time
            }
            
            # Run the workflow
            final_state = self.graph.invoke(initial_state)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            if final_state.get("error"):
                result = ProcessingResult(
                    success=False,
                    document_id=final_state.get("metadata").file_hash if final_state.get("metadata") else "unknown",
                    chunks_created=len(final_state.get("chunks", [])),
                    error_message=final_state.get("error"),
                    processing_time=processing_time,
                    metadata=final_state.get("metadata") if final_state.get("metadata") else DocumentMetadata(
                        filename=file_path.name,
                        file_size=0,
                        num_pages=0
                    )
                )
            else:
                result = ProcessingResult(
                    success=True,
                    document_id=final_state.get("metadata").file_hash if final_state.get("metadata") else "unknown",
                    chunks_created=len(final_state.get("chunks", [])),
                    error_message=None,
                    processing_time=processing_time,
                    metadata=final_state.get("metadata")
                )
            
            logger.info(f"Document processing completed: {result.success}")
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in document processing: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ProcessingResult(
                success=False,
                document_id="unknown",
                chunks_created=0,
                error_message=str(e),
                processing_time=time.time() - start_time,
                metadata=DocumentMetadata(
                    filename=file_path.name,
                    file_size=0,
                    num_pages=0
                )
            ) 