#!/usr/bin/env python3
"""
PDF Document Ingestion System
A clean system for parsing PDFs, chunking them, generating embeddings, and storing in Pinecone.
"""

import argparse
import logging
import sys
from pathlib import Path
import asyncio

from app.service.ingest_service import IngestService
from app.interface.console_chat import ConsoleChatInterface
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ingestion.log')
    ]
)

logger = logging.getLogger(__name__)


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="PDF Document Ingestion System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py process-file document.pdf
  python main.py process-directory /path/to/documents
  python main.py process-directory /path/to/documents --recursive
  python main.py search "your search query"
  python main.py stats
  python main.py chat
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process single file
    file_parser = subparsers.add_parser('process-file', help='Process a single PDF file')
    file_parser.add_argument('file_path', type=str, help='Path to the PDF file')
    
    # Process directory
    dir_parser = subparsers.add_parser('process-directory', help='Process all PDF files in a directory')
    dir_parser.add_argument('directory_path', type=str, help='Path to the directory')
    dir_parser.add_argument('--recursive', action='store_true', help='Process subdirectories recursively')
    
    # Search
    search_parser = subparsers.add_parser('search', help='Search for documents')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    
    # Stats
    stats_parser = subparsers.add_parser('stats', help='Get system statistics')
    
    # Delete document
    delete_parser = subparsers.add_parser('delete', help='Delete a document by file hash')
    delete_parser.add_argument('file_hash', type=str, help='File hash of the document to delete')
    
    # Chat
    chat_parser = subparsers.add_parser('chat', help='Start interactive chatbot')
    
    return parser


def process_file(file_path: str) -> None:
    """Process a single PDF file."""
    logger.info(f"Processing file: {file_path}")
    
    ingest_service = IngestService()
    result = ingest_service.process_single_document(Path(file_path))
    
    if result.success:
        logger.info(f"✅ Successfully processed {file_path}")
        logger.info(f"   - Chunks created: {result.chunks_created}")
        logger.info(f"   - Processing time: {result.processing_time:.2f}s")
        logger.info(f"   - Document ID: {result.document_id}")
    else:
        logger.error(f"❌ Failed to process {file_path}")
        logger.error(f"   - Error: {result.error_message}")


def process_directory(directory_path: str, recursive: bool = False) -> None:
    """Process all PDF files in a directory."""
    logger.info(f"Processing directory: {directory_path} (recursive: {recursive})")
    
    ingest_service = IngestService()
    results = ingest_service.process_directory(Path(directory_path), recursive)
    
    if not results:
        logger.warning("No PDF files found in the directory.")
        return
    
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    logger.info(f"📊 Processing Summary:")
    logger.info(f"   - Total files: {len(results)}")
    logger.info(f"   - Successful: {successful}")
    logger.info(f"   - Failed: {failed}")
    
    for result in results:
        if result.success:
            logger.info(f"   ✅ {result.metadata.filename}: {result.chunks_created} chunks")
        else:
            logger.error(f"   ❌ {result.metadata.filename}: {result.error_message}")


def search_documents(query: str, top_k: int = 20) -> None:
    """Search for documents."""
    logger.info(f"Searching for: {query}")
    
    ingest_service = IngestService()
    results = ingest_service.search_documents(query, top_k)
    
    if results:
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            logger.info(f"\n{i}. Score: {result['score']:.4f}")
            logger.info(f"   File: {result['metadata'].get('filename', 'Unknown')}")
            logger.info(f"   Chunk: {result['metadata'].get('chunk_index', 'Unknown')}")
            content = result['metadata'].get('content', 'No content available')
            logger.info(f"   Content: {content[:200]}...")
    else:
        logger.info("No results found")


def show_stats() -> None:
    """Show system statistics."""
    logger.info("Getting system statistics...")
    
    ingest_service = IngestService()
    stats = ingest_service.get_processing_stats()
    
    if "error" in stats:
        logger.error(f"❌ Error getting stats: {stats['error']}")
        return
    
    logger.info("📈 System Statistics:")
    logger.info(f"   - Embedding Model: {stats.get('embedding_model', 'Unknown')}")
    logger.info(f"   - Chunk Size: {stats.get('chunk_size', 'Unknown')}")
    logger.info(f"   - Chunk Overlap: {stats.get('chunk_overlap', 'Unknown')}")
    
    vector_stats = stats.get("vector_database_stats", {})
    logger.info(f"   - Total Vectors: {vector_stats.get('total_vector_count', 'Unknown')}")
    logger.info(f"   - Vector Dimension: {vector_stats.get('dimension', 'Unknown')}")
    logger.info(f"   - Index Fullness: {vector_stats.get('index_fullness', 'Unknown')}")


def delete_document(file_hash: str) -> None:
    """Delete a document by file hash."""
    logger.info(f"Deleting document with hash: {file_hash}")
    
    ingest_service = IngestService()
    result = ingest_service.delete_document(file_hash)
    
    if "error" in result:
        logger.error(f"❌ Failed to delete document: {result['error']}")
    else:
        deleted_count = result.get('deleted_count', 0)
        logger.info(f"✅ Successfully deleted {deleted_count} vectors")


def start_chat() -> None:
    """Start the interactive chatbot."""
    logger.info("Starting interactive chatbot...")
    
    chat_interface = ConsoleChatInterface()
    chat_interface.run()


def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'process-file':
            process_file(args.file_path)
        elif args.command == 'process-directory':
            process_directory(args.directory_path, args.recursive)
        elif args.command == 'search':
            search_documents(args.query, args.top_k)
        elif args.command == 'stats':
            show_stats()
        elif args.command == 'delete':
            delete_document(args.file_hash)
        elif args.command == 'chat':
            start_chat()
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 