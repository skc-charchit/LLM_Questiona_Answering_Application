import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

class EmbeddingManager:
    """Class for managing embeddings and vector stores."""

    def __init__(self, model_name=None):
        """
        Initialize with embedding model.

        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def create_vector_store(self, documents):
        """
        Create a vector store from documents.

        Args:
            documents: List of Document objects

        Returns:
            FAISS vector store
        """
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        return vectorstore

    def save_vector_store(self, vectorstore, directory="vectorstore"):
        """
        Save a vector store to disk.

        Args:
            vectorstore: FAISS vector store
            directory: Directory to save the vector store

        Returns:
            Path to the saved vector store
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save vector store
        vectorstore.save_local(directory)

        return directory

    def load_vector_store(self, directory="vectorstore"):
        """
        Load a vector store from disk.

        Args:
            directory: Directory containing the vector store

        Returns:
            FAISS vector store
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Vector store directory {directory} not found")

        vectorstore = FAISS.load_local(directory, self.embeddings)
        return vectorstore

    def print_vector_store_info(self, vectorstore, num_documents=None):
        """
        Print information about a vector store.

        Args:
            vectorstore: FAISS vector store
            num_documents: Number of documents in the vector store
        """
        print("\n" + "="*50)
        print("VECTOR STORE INFORMATION")
        print("="*50)
        print(f"Vector store type: FAISS (local)")
        print(f"Embedding model: {self.model_name}")
        if num_documents:
            print(f"Documents processed: {num_documents}")
        print("="*50)

# Example usage
if __name__ == "__main__":
    import sys
    import os

    # Add parent directory to path to import from sibling packages
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from src.tasks.loader import DocumentLoader

    # Load and process documents
    doc_loader = DocumentLoader()
    documents = doc_loader.load_pdf(os.path.join("..", "data", "stats.pdf"), verbose=True)
    split_docs = doc_loader.split_documents(documents)

    # Create and save vector store
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.create_vector_store(split_docs)
    save_dir = embedding_manager.save_vector_store(vectorstore)

    # Print information
    embedding_manager.print_vector_store_info(vectorstore, len(split_docs))
    print(f"Vector store saved at: {save_dir}")
