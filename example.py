#!/usr/bin/env python3
"""
Example usage of the PDF Document Ingestion System
"""

import asyncio
import logging
import sys
from pathlib import Path
from app.service.ingest_service import IngestService

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


def example_single_document():
    """Example: Process a single PDF document."""
    print("=== Single Document Processing Example ===")
    
    # Initialize the service
    ingest_service = IngestService()
    
    # Example file path (replace with your actual PDF file)
    file_path = Path("Profile2.pdf")
    
    if not file_path.exists():
        print(f"⚠️  File {file_path} not found. Please place a PDF file in the current directory.")
        return
    
    # Process the document
    result = ingest_service.process_single_document(file_path)
    
    if result.success:
        print(f"✅ Successfully processed {file_path.name}")
        print(f"   - Chunks created: {result.chunks_created}")
        print(f"   - Processing time: {result.processing_time:.2f}s")
        print(f"   - Document ID: {result.document_id}")
    else:
        print(f"❌ Failed to process {file_path.name}")
        print(f"   - Error: {result.error_message}")


def example_directory_processing():
    """Example: Process all PDF files in a directory."""
    print("\n=== Directory Processing Example ===")
    
    # Initialize the service
    ingest_service = IngestService()
    
    # Example directory path (replace with your actual directory)
    directory_path = Path("documents")
    
    if not directory_path.exists():
        print(f"⚠️  Directory {directory_path} not found. Creating example directory...")
        directory_path.mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory_path}")
        print("   Please place PDF files in this directory and run the example again.")
        return
    
    # Process all PDF files in the directory
    results = ingest_service.process_directory(directory_path)
    
    if not results:
        print("No PDF files found in the directory.")
        return
    
    # Display results
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"📊 Processing Summary:")
    print(f"   - Total files: {len(results)}")
    print(f"   - Successful: {successful}")
    print(f"   - Failed: {failed}")
    
    for result in results:
        if result.success:
            print(f"   ✅ {result.metadata.filename}: {result.chunks_created} chunks")
        else:
            print(f"   ❌ {result.metadata.filename}: {result.error_message}")


def example_search():
    """Example: Search for documents."""
    print("\n=== Document Search Example ===")
    
    # Initialize the service
    ingest_service = IngestService()
    
    # Example search queries
    queries = [
        "Vue.js",
        "Bachelor of Science"
    ]
    
    for query in queries:
        print(f"\n🔍 Searching for: '{query}'")
        print("-" * 50)
        results = ingest_service.search_documents(query, top_k=3)
        
        if results:
            print(f"   Found {len(results)} results:")
            print(f"   Found {results} results:")

            for i, result in enumerate(results, 1):
                print(f"\n   {i}. Score: {result['score']:.4f}")
                print(f"      📄 File: {result['metadata'].get('filename', 'Unknown')}")
                print(f"      📍 Chunk: {result['metadata'].get('chunk_index', 'Unknown')}")
                content = result['metadata'].get('content', 'No content available')
                if content:
                    # Display first 300 characters, truncate if longer
                    display_content = content[:300] + "..." if len(content) > 300 else content
                    print(f"      📝 Content: {display_content}")
                print()
        else:
            print("   No results found")


def example_system_stats():
    """Example: Get system statistics."""
    print("\n=== System Statistics Example ===")
    
    # Initialize the service
    ingest_service = IngestService()
    
    # Get statistics
    stats = ingest_service.get_processing_stats()
    
    if "error" in stats:
        print(f"❌ Error getting stats: {stats['error']}")
        return
    
    print("📈 System Statistics:")
    print(f"   - Embedding Model: {stats.get('embedding_model', 'Unknown')}")
    print(f"   - Chunk Size: {stats.get('chunk_size', 'Unknown')}")
    print(f"   - Chunk Overlap: {stats.get('chunk_overlap', 'Unknown')}")
    
    vector_stats = stats.get("vector_database_stats", {})
    print(f"   - Total Vectors: {vector_stats.get('total_vector_count', 'Unknown')}")
    print(f"   - Vector Dimension: {vector_stats.get('dimension', 'Unknown')}")
    print(f"   - Index Fullness: {vector_stats.get('index_fullness', 'Unknown')}")


async def example_async_processing():
    """Example: Process multiple documents asynchronously."""
    print("\n=== Async Processing Example ===")
    
    # Initialize the service
    ingest_service = IngestService()
    
    # Example file paths (replace with your actual PDF files)
    file_paths = [
        Path("doc1.pdf"),
        Path("doc2.pdf"),
        Path("doc3.pdf")
    ]
    
    # Filter to only existing files
    existing_files = [f for f in file_paths if f.exists()]
    
    if not existing_files:
        print("⚠️  No PDF files found for async processing.")
        print("   Please place PDF files in the current directory and run the example again.")
        return
    
    print(f"🔄 Processing {len(existing_files)} files asynchronously...")
    
    # Process files asynchronously
    results = await ingest_service.process_documents_async(existing_files, max_workers=2)
    
    # Display results
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"📊 Async Processing Summary:")
    print(f"   - Total files: {len(results)}")
    print(f"   - Successful: {successful}")
    print(f"   - Failed: {failed}")


def main():
    """Run all examples."""
    print("🚀 PDF Document Ingestion System - Examples")
    print("=" * 50)
    
    # Check if environment is properly configured
    try:
        from config import settings
        if not settings.openai_api_key or not settings.pinecone_api_key:
            print("⚠️  Please configure your API keys in the .env file before running examples.")
            print("   See README.md for setup instructions.")
            return
    except Exception as e:
        print(f"⚠️  Configuration error: {e}")
        print("   Please check your .env file and configuration.")
        return
    
    # Run examples
    example_single_document()
    #example_directory_processing()
    example_search()
    #example_system_stats()
    
    # Run async example
    asyncio.run(example_async_processing())
    
    print("\n✅ Examples completed!")
    print("📖 For more information, see the README.md file.")


if __name__ == "__main__":
    main() 

