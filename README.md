# PDF Document Ingestion System

A clean, production-ready system for parsing PDF documents, chunking them into manageable pieces, generating embeddings using OpenAI's text-embedding-3-large model, and storing them in Pinecone for semantic search.

## Features

- **PDF Parsing**: Robust PDF text extraction using PyPDF
- **Smart Chunking**: Intelligent text chunking with configurable size and overlap
- **High-Quality Embeddings**: Uses OpenAI's text-embedding-3-large model for superior semantic understanding
- **Vector Storage**: Pinecone integration for scalable vector search
- **LangGraph Workflow**: Clean, orchestrated processing pipeline with error handling
- **Async Processing**: Support for concurrent document processing
- **CLI Interface**: Easy-to-use command-line interface
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Error Handling**: Robust error handling and recovery
- **File Deduplication**: Automatic handling of duplicate files using file hashing

## Architecture

The system is built with a clean, modular architecture:

```
app/
├── core/                 # Core processing components
│   ├── models.py        # Pydantic data models
│   ├── pdf_parser.py    # PDF parsing and chunking
│   ├── embedding_service.py  # OpenAI embedding generation
│   └── workflow.py      # LangGraph workflow orchestration
├── service/             # Service layer
│   └── ingest_service.py # Main ingestion service
├── vectorstore/         # Vector database components
│   └── pinecone_service.py # Pinecone operations
└── api/                 # API endpoints (future)
```

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key and environment

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-sample
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Pinecone Configuration
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment_here
   PINECONE_INDEX_NAME=pdf-documents
   
   # Embedding Model Configuration
   EMBEDDING_MODEL=text-embedding-3-large
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

## Usage

### Command Line Interface

The system provides a comprehensive CLI for all operations:

#### Process a Single PDF File
```bash
python main.py process-file path/to/document.pdf
```

#### Process All PDFs in a Directory
```bash
python main.py process-directory /path/to/documents
```

#### Process PDFs Recursively (including subdirectories)
```bash
python main.py process-directory /path/to/documents --recursive
```

#### Search for Documents
```bash
python main.py search "your search query"
python main.py search "machine learning algorithms" --top-k 10
```

#### View System Statistics
```bash
python main.py stats
```

#### Delete a Document
```bash
python main.py delete <file_hash>
```

### Programmatic Usage

You can also use the system programmatically:

```python
from pathlib import Path
from app.service.ingest_service import IngestService

# Initialize the service
ingest_service = IngestService()

# Process a single document
result = ingest_service.process_single_document(Path("document.pdf"))
print(f"Processed {result.chunks_created} chunks")

# Process multiple documents
results = ingest_service.process_multiple_documents([
    Path("doc1.pdf"),
    Path("doc2.pdf")
])

# Search for documents
search_results = ingest_service.search_documents("machine learning", top_k=5)

# Get system statistics
stats = ingest_service.get_processing_stats()
```

## Configuration

The system is highly configurable through the `config.py` file and environment variables:

### Chunking Configuration
- `CHUNK_SIZE`: Size of text chunks (default: 1000 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200 characters)

### Embedding Configuration
- `EMBEDDING_MODEL`: OpenAI embedding model (default: text-embedding-3-large)
- `BATCH_SIZE`: Number of embeddings to generate in parallel (default: 100)

### File Processing
- `MAX_FILE_SIZE_MB`: Maximum file size limit (default: 50MB)

## LangGraph Workflow

The system uses LangGraph to orchestrate the document processing pipeline:

1. **PDF Parsing**: Extract text and metadata from PDF
2. **Text Chunking**: Split text into manageable chunks
3. **Embedding Generation**: Generate embeddings for each chunk
4. **Vector Storage**: Upsert embeddings to Pinecone

Each step includes comprehensive error handling and logging.

## Error Handling

The system includes robust error handling:

- **File Validation**: Checks file existence, type, and size
- **PDF Processing**: Handles corrupted or unreadable PDFs
- **API Errors**: Graceful handling of OpenAI and Pinecone API errors
- **Network Issues**: Retry logic for transient failures
- **Memory Management**: Efficient processing of large documents

## Logging

The system provides comprehensive logging:

- **Console Output**: Real-time processing status
- **File Logging**: Detailed logs saved to `ingestion.log`
- **Structured Logging**: JSON-formatted logs for production monitoring

## Performance Considerations

- **Batch Processing**: Embeddings are generated in batches for efficiency
- **Async Support**: Concurrent processing of multiple documents
- **Memory Efficient**: Streaming processing for large files
- **Caching**: Intelligent caching of embeddings and metadata

## Security

- **API Key Management**: Secure handling of API keys through environment variables
- **File Validation**: Strict validation of input files
- **Error Sanitization**: Sensitive information is not logged

## Monitoring and Maintenance

### Health Checks
```bash
python main.py stats
```

### Index Management
The system automatically creates and manages Pinecone indexes. You can monitor index health through the stats command.

### Cleanup
To remove documents from the vector store:
```bash
python main.py delete <file_hash>
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your OpenAI and Pinecone API keys are correctly set in the `.env` file
2. **File Not Found**: Verify the file path and ensure the file exists
3. **Memory Issues**: For large documents, consider reducing chunk size
4. **Rate Limiting**: The system includes built-in rate limiting for API calls

### Debug Mode
Enable debug logging by modifying the logging level in `main.py`:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `ingestion.log`
3. Open an issue on GitHub

## Roadmap

- [ ] Web API interface
- [ ] Support for additional document formats (DOCX, TXT)
- [ ] Advanced search filters
- [ ] Document versioning
- [ ] Multi-language support
- [ ] Real-time processing dashboard 