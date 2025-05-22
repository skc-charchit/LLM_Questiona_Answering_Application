"""
Together.ai embeddings implementation for LangChain.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
import requests

# Load environment variables
load_dotenv()

class TogetherEmbeddings(Embeddings):
    """Together.ai embeddings implementation for LangChain."""

    def __init__(
        self,
        model_name: str = "togethercomputer/m2-bert-80M-8k-retrieval",
        api_key: Optional[str] = None,
        dimensions: int = 768,
    ):
        """
        Initialize Together.ai embeddings.

        Args:
            model_name: Name of the embedding model to use
            api_key: Together.ai API key
            dimensions: Dimensions of the embeddings
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment variables or parameters")

        self.dimensions = dimensions
        self.base_url = "https://api.together.xyz/v1/embeddings"

        print(f"Using Together.ai embedding model: {self.model_name}")

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "input": text
        }

        response = requests.post(self.base_url, headers=headers, json=data)

        if response.status_code != 200:
            raise ValueError(f"Error from Together.ai API: {response.text}")

        result = response.json()
        embedding = result["data"][0]["embedding"]

        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query.

        Args:
            text: Query to embed

        Returns:
            Query embedding
        """
        return self._get_embedding(text)

class EmbeddingManager:
    """Class for managing embeddings and vector stores."""

    def __init__(self, embeddings=None, model_name=None, api_key=None):
        """
        Initialize with embedding model.

        Args:
            embeddings: Pre-configured embeddings instance (takes precedence if provided)
            model_name: Name of the embedding model to use (used if embeddings not provided)
            api_key: Together.ai API key
        """
        # Use provided embeddings or create new ones
        if embeddings:
            self.embeddings = embeddings
            print(f"Using externally provided embeddings")
        else:
            # Get model name from environment or parameter
            self.model_name = model_name or os.getenv(
                "TOGETHER_EMBEDDING_MODEL",
                "togethercomputer/m2-bert-80M-8k-retrieval"
            )

            # Create embeddings
            self.embeddings = TogetherEmbeddings(
                model_name=self.model_name,
                api_key=api_key
            )
            print(f"Created embeddings with model: {self.model_name}")

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
        print(f"Embedding model: {self.model_name if hasattr(self, 'model_name') else 'Custom'}")
        if num_documents:
            print(f"Documents processed: {num_documents}")
        print("="*50)

