# SOP_GMP_retriever.py
import os
from pinecone import Pinecone
import cohere
from dotenv import load_dotenv

# --- Initialization ---
# Load environment variables from .env file
load_dotenv()

class SOPGMPRetriever:
    """
    A class to retrieve SOP and GMP documents using Cohere embeddings and Pinecone vector database.
    """
    
    def __init__(self):
        """Initialize the retriever with Cohere and Pinecone clients."""
        # Initialize Cohere client
        try:
            cohere_api_key = os.getenv("COHERE_API_KEY")
            self.co = cohere.Client(cohere_api_key)
        except Exception as e:
            raise Exception(f"Error initializing Cohere client: {e}")
        
        # Initialize Pinecone client
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            self.pc = Pinecone(api_key=pinecone_api_key)
        except Exception as e:
            raise Exception(f"Error initializing Pinecone: {e}")
        
        # Configuration
        self.index_name = "gsk-qa-copilot"
        self.embedding_model = "embed-english-v3.0"
        
        # Connect to Pinecone index
        try:
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            raise Exception(f"Error connecting to Pinecone index '{self.index_name}': {e}")
    
    def query_sops(self, query_text, top_k=5):
        """
        Query SOP and GMP documents using Cohere embeddings and Pinecone vector search.
        
        Args:
            query_text (str): The query text to search for
            top_k (int): Number of top similar documents to return (default: 5)
            
        Returns:
            list: List of dictionaries containing:
                - text: The document chunk text
                - source_document: The source document filename
                - score: The similarity score from Pinecone
        """
        try:
            # Generate embedding for the query using Cohere
            query_embedding = self._get_query_embedding(query_text)
            
            if not query_embedding:
                return []
            
            # Query Pinecone vector database
            query_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            # Process and format results
            results = []
            for match in query_results.matches:
                result = {
                    "text": match.metadata.get("text", ""),
                    "source_document": match.metadata.get("source_document", ""),
                    "score": match.score
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error querying SOPs: {e}")
            return []
    
    def _get_query_embedding(self, query_text):
        """
        Generate embedding for query text using Cohere.
        
        Args:
            query_text (str): The text to embed
            
        Returns:
            list: The embedding vector or None if error
        """
        try:
            response = self.co.embed(
                texts=[query_text],
                model=self.embedding_model,
                input_type="search_query"  # Use 'search_query' for querying
            )
            return response.embeddings[0] if response.embeddings else None
        except Exception as e:
            print(f"Error getting query embedding from Cohere: {e}")
            return None
    
    def get_index_stats(self):
        """
        Get statistics about the Pinecone index.
        
        Returns:
            dict: Index statistics
        """
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}

    def interactive_query(self):
        """
        Interactive command-line interface for querying SOPs.
        """
        print("🔍 SOP & GMP Document Retriever")
        print("=" * 50)
        print("Enter your queries to search through SOP and GMP documents.")
        print("Type 'quit', 'exit', or 'q' to stop.")
        print("Type 'stats' to see index statistics.")
        print("Type 'help' for more options.")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                query = input("\n📝 Enter your query: ").strip()
                
                # Handle special commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'stats':
                    stats = self.get_index_stats()
                    print(f"\n📊 Index Statistics:")
                    print(f"   Total vectors: {stats.get('total_vector_count', 'N/A')}")
                    print(f"   Dimension: {stats.get('dimension', 'N/A')}")
                    print(f"   Index fullness: {stats.get('index_fullness', 'N/A')}")
                    continue
                elif query.lower() == 'help':
                    print("\n💡 Available commands:")
                    print("   - Enter any text to search SOPs and GMP documents")
                    print("   - 'stats' - Show index statistics")
                    print("   - 'quit', 'exit', or 'q' - Exit the program")
                    print("   - 'help' - Show this help message")
                    continue
                elif not query:
                    print("❌ Please enter a query.")
                    continue
                
                # Ask for number of results
                try:
                    top_k_input = input("🔢 Number of results to return (default 5): ").strip()
                    top_k = int(top_k_input) if top_k_input else 5
                except ValueError:
                    top_k = 5
                    print("⚠️  Invalid number, using default of 5 results.")
                
                # Perform the query
                print(f"\n🔍 Searching for: '{query}'")
                print("⏳ Processing...")
                
                results = self.query_sops(query, top_k=top_k)
                
                if not results:
                    print("❌ No results found. Try a different query.")
                    continue
                
                # Display results
                print(f"\n✅ Found {len(results)} results:")
                print("=" * 60)
                
                for i, result in enumerate(results, 1):
                    print(f"\n📄 Result {i}:")
                    print(f"   📊 Similarity Score: {result['score']:.4f}")
                    print(f"   📁 Source Document: {result['source_document']}")
                    print(f"   📝 Content Preview:")
                    # Display text with better formatting
                    text_preview = result['text'][:300]
                    if len(result['text']) > 300:
                        text_preview += "..."
                    
                    # Format text with proper indentation
                    for line in text_preview.split('\n'):
                        print(f"      {line}")
                
                print("\n" + "=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again.")

# Example usage and interactive mode
if __name__ == "__main__":
    try:
        # Initialize the retriever
        print("🚀 Initializing SOP & GMP Retriever...")
        retriever = SOPGMPRetriever()
        print("✅ Retriever initialized successfully!")
        
        # Start interactive mode
        retriever.interactive_query()
        
    except Exception as e:
        print(f"❌ Failed to initialize retriever: {e}")
        print("Please check your environment variables (COHERE_API_KEY, PINECONE_API_KEY)")
