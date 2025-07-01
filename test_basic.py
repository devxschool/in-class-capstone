#!/usr/bin/env python3
"""
Basic tests for the PDF Document Ingestion System
"""

import unittest
from pathlib import Path
import tempfile
import os

from app.core.models import DocumentChunk, DocumentMetadata, ProcessingResult
from app.core.pdf_parser import PDFParser
from app.core.embedding_service import EmbeddingService
from config import settings


class TestBasicComponents(unittest.TestCase):
    """Basic tests for core components."""
    
    def setUp(self):
        """Set up test environment."""
        # Check if API keys are configured
        if not settings.openai_api_key or not settings.pinecone_api_key:
            self.skipTest("API keys not configured")
    
    def test_pdf_parser_initialization(self):
        """Test PDF parser initialization."""
        parser = PDFParser()
        self.assertIsNotNone(parser)
        self.assertIsNotNone(parser.text_splitter)
    
    def test_embedding_service_initialization(self):
        """Test embedding service initialization."""
        service = EmbeddingService()
        self.assertIsNotNone(service)
        self.assertIsNotNone(service.embeddings)
    
    def test_document_chunk_creation(self):
        """Test DocumentChunk model creation."""
        chunk = DocumentChunk(
            content="Test content",
            chunk_index=0,
            source_file="test.pdf"
        )
        self.assertEqual(chunk.content, "Test content")
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.source_file, "test.pdf")
        self.assertIsNotNone(chunk.id)
    
    def test_document_metadata_creation(self):
        """Test DocumentMetadata model creation."""
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_size=1024,
            num_pages=5
        )
        self.assertEqual(metadata.filename, "test.pdf")
        self.assertEqual(metadata.file_size, 1024)
        self.assertEqual(metadata.num_pages, 5)
        self.assertIsNotNone(metadata.created_at)
    
    def test_processing_result_creation(self):
        """Test ProcessingResult model creation."""
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_size=1024,
            num_pages=5
        )
        
        result = ProcessingResult(
            success=True,
            document_id="test_id",
            chunks_created=10,
            processing_time=1.5,
            metadata=metadata
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.document_id, "test_id")
        self.assertEqual(result.chunks_created, 10)
        self.assertEqual(result.processing_time, 1.5)
        self.assertEqual(result.metadata, metadata)


class TestConfiguration(unittest.TestCase):
    """Test configuration settings."""
    
    def test_settings_loaded(self):
        """Test that settings are properly loaded."""
        self.assertIsNotNone(settings.embedding_model)
        self.assertIsNotNone(settings.chunk_size)
        self.assertIsNotNone(settings.chunk_overlap)
        self.assertIsNotNone(settings.max_file_size_mb)
    
    def test_embedding_model_setting(self):
        """Test embedding model setting."""
        self.assertEqual(settings.embedding_model, "text-embedding-3-large")
    
    def test_chunk_settings(self):
        """Test chunk size and overlap settings."""
        self.assertGreater(settings.chunk_size, 0)
        self.assertGreaterEqual(settings.chunk_overlap, 0)
        self.assertLess(settings.chunk_overlap, settings.chunk_size)


class TestFileValidation(unittest.TestCase):
    """Test file validation logic."""
    
    def test_pdf_parser_file_validation(self):
        """Test PDF parser file validation."""
        parser = PDFParser()
        
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            parser._validate_file(Path("non_existent.pdf"))
        
        # Test with non-PDF file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_file = Path(f.name)
        
        try:
            with self.assertRaises(ValueError):
                parser._validate_file(temp_file)
        finally:
            os.unlink(temp_file)


def run_tests():
    """Run all tests."""
    print("🧪 Running basic tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBasicComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestFileValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"   - Tests run: {result.testsRun}")
    print(f"   - Failures: {len(result.failures)}")
    print(f"   - Errors: {len(result.errors)}")
    print(f"   - Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 