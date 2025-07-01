import os
import hashlib
from typing import List, Optional
from pathlib import Path
import logging

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.models import DocumentChunk, DocumentMetadata
from config import settings

logger = logging.getLogger(__name__)


class PDFParser:
    """Handles PDF document parsing and text extraction."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of the file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _validate_file(self, file_path: Path) -> None:
        """Validate the PDF file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.suffix.lower() == '.pdf':
            raise ValueError(f"File must be a PDF: {file_path}")
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            raise ValueError(f"File size {file_size_mb:.2f}MB exceeds maximum {settings.max_file_size_mb}MB")
    
    def extract_text_from_pdf(self, file_path: Path) -> tuple[str, DocumentMetadata]:
        """Extract text from PDF and return with metadata."""
        self._validate_file(file_path)
        
        try:
            reader = PdfReader(file_path)
            text_content = ""
            
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text.strip():
                    text_content += f"\n\n--- Page {page_num} ---\n\n{page_text}"
            
            if not text_content.strip():
                raise ValueError("No text content extracted from PDF")
            
            metadata = DocumentMetadata(
                filename=file_path.name,
                file_size=file_path.stat().st_size,
                num_pages=len(reader.pages),
                file_hash=self._calculate_file_hash(file_path)
            )
            
            logger.info(f"Successfully extracted text from {file_path.name} ({len(reader.pages)} pages)")
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise
    
    def chunk_text(self, text: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Split text into chunks using LangChain text splitter."""
        try:
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(text)
            
            # Create DocumentChunk objects
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    content=chunk_text.strip(),
                    chunk_index=i,
                    source_file=metadata.filename,
                    metadata={
                        "filename": metadata.filename,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "file_hash": metadata.file_hash
                    }
                )
                chunks.append(chunk)
            
            metadata.total_chunks = len(chunks)
            logger.info(f"Created {len(chunks)} chunks from {metadata.filename}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise
    
    def parse_pdf(self, file_path: Path) -> tuple[List[DocumentChunk], DocumentMetadata]:
        """Complete PDF parsing pipeline: extract text and chunk it."""
        text_content, metadata = self.extract_text_from_pdf(file_path)
        chunks = self.chunk_text(text_content, metadata)
        return chunks, metadata 