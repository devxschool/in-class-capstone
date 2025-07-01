from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_index: int
    page_number: Optional[int] = None
    source_file: str


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    filename: str
    file_size: int
    num_pages: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    total_chunks: int = 0
    file_hash: Optional[str] = None


class ProcessingResult(BaseModel):
    """Result of document processing."""
    success: bool
    document_id: str
    chunks_created: int
    error_message: Optional[str] = None
    processing_time: float
    metadata: DocumentMetadata


class EmbeddingResult(BaseModel):
    """Result of embedding generation."""
    chunk_id: str
    embedding: List[float]
    metadata: Dict[str, Any] 