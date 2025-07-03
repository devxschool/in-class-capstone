import logging
import sys
from typing import List, Dict, Any
from pathlib import Path

from app.service.chatbot_service import ChatbotService
from config import settings

logger = logging.getLogger(__name__)


class ConsoleChatInterface:
    """Console-based chat interface for the RAG chatbot."""
    
    def __init__(self):
        self.chatbot = ChatbotService()
        self.chat_history: List[Dict[str, str]] = []
        
    def _print_welcome(self):
        """Print welcome message and instructions."""
        print("Welcome to the RAG Chatbot!")
        print("=" * 50)
        print("This chatbot can answer questions based on your uploaded documents.")
        print("Commands:")
        print("  /help     - Show this help message")
        print("  /stats    - Show system statistics")
        print("  /clear    - Clear chat history")
        print("  /quit     - Exit the chatbot")
        print("  /sources  - Show sources for the last response")
        print("=" * 50)
        print()
    
    def _print_stats(self):
        """Print system statistics."""
        print("\n📊 System Statistics:")
        print("-" * 30)
        
        try:
            stats = self.chatbot.get_chat_stats()
            
            if "error" in stats:
                print(f"❌ Error getting stats: {stats['error']}")
                return
            
            print(f"🤖 Model: {stats.get('model', 'Unknown')}")
            print(f"🤖 Embedding Model: {stats.get('embedding_model', 'Unknown')}")
            
            vector_stats = stats.get('vector_database_stats', {})
            print(f"📏 Total Documents: {vector_stats.get('total_vector_count', 0)}")
            print(f"📏 Vector Dimension: {vector_stats.get('dimension', 'Unknown')}")
            print(f"📈 Index Fullness: {vector_stats.get('index_fullness', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    def _clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        print("🧹 Chat history cleared!")
        print()
    
    def _show_sources(self, last_response: Dict[str, Any]):
        """Show sources for the last response."""
        sources = last_response.get('sources', [])
        
        if not sources:
            print("No sources used in the last response.")
            return
        
        print(f"\n📚 Sources used ({len(sources)} documents):")
        print("-" * 40)
        
        for i, source in enumerate(sources, 1):
            print(f"{i}. 📄 {source['filename']} (Chunk {source['chunk_index']})")
            print(f"   💡 {source['content']}")
            print()
    
    def _process_user_input(self, user_input: str) -> bool:
        """Process user input and return True if should continue."""
        user_input = user_input.strip()
        
        if not user_input:
            return True
        
        # Handle commands
        if user_input.startswith('/'):
            command = user_input.lower()
            
            if command == '/help':
                self._print_welcome()
            elif command == '/stats':
                self._print_stats()
            elif command == '/clear':
                self._clear_history()
            elif command == '/quit':
                print("👋 Goodbye!")
                return False
            elif command == '/sources':
                if hasattr(self, 'last_response'):
                    self._show_sources(self.last_response)
                else:
                    print("📚 No previous response to show sources for.")
            else:
                print(f"❓ Unknown command: {user_input}")
                print("Type /help for available commands.")
            
            return True
        
        # Process regular message
        print(f"\n🤔 Processing your question...")
        
        try:
            # Get response from chatbot
            response = self.chatbot.chat(user_input, self.chat_history)
            
            # Store response for potential source viewing
            self.last_response = response
            
            # Print response
            print(f"\n🤖 Assistant: {response['response']}")
            
            # Show context info
            if response.get('context_used', 0) > 0:
                print(f"\n📚 Used {response['context_used']} document chunks for context")
                print(f"💡 Type /sources to see the source documents")
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": response['response']})
            
            # Keep only last 10 messages to prevent context from getting too long
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Error processing user input: {str(e)}")
        
        print()
        return True
    
    def run(self):
        """Run the console chat interface."""
        try:
            # Check configuration
            if not settings.openai_api_key:
                print("❌ OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
                return
            
            if not settings.pinecone_api_key:
                print("❌ Pinecone API key not configured. Please set PINECONE_API_KEY in your .env file.")
                return
            
            # Print welcome message
            self._print_welcome()
            
            # Show initial stats
            self._print_stats()
            
            # Main chat loop
            while True:
                try:
                    user_input = input("💬 You: ")
                    if not self._process_user_input(user_input):
                        break
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n\n👋 Goodbye!")
                    break
                    
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            logger.error(f"Fatal error in console chat: {str(e)}")


def main():
    """Main entry point for console chat."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('chatbot.log')
        ]
    )
    
    # Run the chat interface
    chat_interface = ConsoleChatInterface()
    chat_interface.run()


if __name__ == "__main__":
    main() 