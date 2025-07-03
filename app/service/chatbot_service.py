import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.embedding_service import EmbeddingService
from app.vectorstore.pinecone_service import PineconeService
from app.core.models import DocumentChunk
from config import settings

logger = logging.getLogger(__name__)


class ChatbotService:
    """RAG-powered chatbot service using LangChain and LangGraph."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            openai_api_key=settings.openai_api_key
        )
        self.embedding_service = EmbeddingService()
        self.pinecone_service = PineconeService()
        
        # Define the system prompt
        self.system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from documents. 

When answering questions:
1. Use only the information provided in the context
2. If the context doesn't contain enough information to answer the question, say so
3. Be concise but thorough
4. Cite the source documents when possible
5. If you're unsure about something, acknowledge the uncertainty

Context from documents:
{context}

Current conversation:
{chat_history}

User question: {question}

Please provide a helpful answer based on the context above."""

    def _retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context from the vector store."""
        try:
            # Create a query chunk
            query_chunk = DocumentChunk(
                content=query,
                chunk_index=0,
                source_file="query"
            )
            
            # Generate embedding for the query
            query_embedding_result = self.embedding_service.generate_embeddings([query_chunk])[0]
            
            # Search in Pinecone
            search_results = self.pinecone_service.search_similar(
                query_embedding_result.embedding,
                top_k=top_k
            )
            
            logger.info(f"Retrieved {len(search_results)} relevant documents for query")
            return search_results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context string."""
        if not search_results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            metadata = result.get('metadata', {})
            content = metadata.get('content', 'No content available')
            filename = metadata.get('filename', 'Unknown file')
            chunk_index = metadata.get('chunk_index', 'Unknown')
            
            context_parts.append(f"Document {i}: {filename} (Chunk {chunk_index})")
            context_parts.append(f"Content: {content}")
            context_parts.append("---")
        
        return "\n".join(context_parts)

    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Format chat history for the prompt."""
        if not chat_history:
            return "No previous conversation."
        
        history_parts = []
        for message in chat_history:
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            history_parts.append(f"{role}: {content}")
        
        return "\n".join(history_parts)

    def chat(self, user_message: str, chat_history: Optional[List[Dict[str, str]]] = None, 
             top_k: int = 5) -> Dict[str, Any]:
        """Process a user message and return a response."""
        try:
            if chat_history is None:
                chat_history = []
            
            logger.info(f"Processing user message: {user_message[:100]}...")
            
            # Step 1: Retrieve relevant context
            search_results = self._retrieve_relevant_context(user_message, top_k)
            
            # Step 2: Format context and chat history
            context = self._format_context(search_results)
            formatted_history = self._format_chat_history(chat_history)
            
            # Step 3: Create the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt)
            ])
            
            # Step 4: Generate response
            messages = prompt.format_messages(
                context=context,
                chat_history=formatted_history,
                question=user_message
            )
            
            response = self.llm.invoke(messages)
            
            # Step 5: Prepare result
            result = {
                "response": response.content,
                "context_used": len(search_results),
                "sources": [
                    {
                        "filename": result.get('metadata', {}).get('filename', 'Unknown'),
                        "chunk_index": result.get('metadata', {}).get('chunk_index', 'Unknown'),
                        "content": result.get('metadata', {}).get('content', '')[:200] + "..."
                    }
                    for result in search_results
                ]
            }
            
            logger.info(f"Generated response with {len(search_results)} context sources")
            return result
            
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "context_used": 0,
                "sources": [],
                "error": str(e)
            }

    def get_chat_stats(self) -> Dict[str, Any]:
        """Get statistics about the chatbot system."""
        try:
            vector_stats = self.pinecone_service.get_index_stats()
            return {
                "model": "gpt-4o",
                "embedding_model": settings.embedding_model,
                "vector_database_stats": vector_stats,
                "total_documents": vector_stats.get('total_vector_count', 0)
            }
        except Exception as e:
            logger.error(f"Error getting chat stats: {str(e)}")
            return {"error": str(e)} 