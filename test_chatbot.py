#!/usr/bin/env python3
"""
Test script for the RAG chatbot
"""

import logging
from app.service.chatbot_service import ChatbotService
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_chatbot():
    """Test the chatbot functionality."""
    print("🧪 Testing RAG Chatbot...")
    
    # Check configuration
    if not settings.openai_api_key:
        print("❌ OpenAI API key not configured")
        return
    
    if not settings.pinecone_api_key:
        print("❌ Pinecone API key not configured")
        return
    
    # Initialize chatbot
    chatbot = ChatbotService()
    
    # Test queries
    test_queries = [
        "What documents do you have access to?",
        "Can you summarize the main topics in the documents?",
        "What are the key findings in the documents?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: {query}")
        print("-" * 50)
        
        try:
            response = chatbot.chat(query)
            print(f"🤖 Response: {response['response']}")
            print(f"📚 Sources used: {response['context_used']}")
            
            if response.get('sources'):
                print("📄 Sources:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"   {i}. {source['filename']} (Chunk {source['chunk_index']})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    # Test stats
    print("📊 Chatbot Statistics:")
    stats = chatbot.get_chat_stats()
    if "error" not in stats:
        print(f"   Model: {stats.get('model')}")
        print(f"   Total Documents: {stats.get('total_documents', 0)}")
    else:
        print(f"   Error getting stats: {stats['error']}")


if __name__ == "__main__":
    test_chatbot() 