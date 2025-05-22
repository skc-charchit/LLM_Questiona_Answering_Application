import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

class DocumentLoader:
    """Class for loading and processing PDF documents."""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """Initialize with text splitting parameters."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_pdf(self, pdf_path, verbose=False):
        """
        Load a PDF file from a path.

        Args:
            pdf_path: Path to the PDF file
            verbose: Whether to print information about the loaded PDF

        Returns:
            List of Document objects
        """
        # Convert to absolute path if relative
        if not os.path.isabs(pdf_path):
            base_dir = Path(__file__).parent.parent.parent  # Go up to project root
            pdf_path = os.path.join(base_dir, pdf_path)

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if verbose and documents:
            self._print_document_info(documents)

        return documents

    def load_online_pdf(self, pdf_url, verbose=False):
        """
        Load a PDF from a URL.

        Args:
            pdf_url: URL to the PDF file
            verbose: Whether to print information about the loaded PDF

        Returns:
            List of Document objects
        """
        loader = OnlinePDFLoader(pdf_url)
        documents = loader.load()

        if verbose and documents:
            self._print_document_info(documents)

        return documents

    def split_documents(self, documents):
        """
        Split documents into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of split Document objects
        """
        return self.text_splitter.split_documents(documents)

    def _print_document_info(self, documents):
        """Print information about the loaded documents."""
        print("\n" + "="*50)
        print("PDF DOCUMENT INFORMATION")
        print("="*50)
        print(f"Total pages loaded: {len(documents)}")
        print("-"*50)

        if documents:
            doc = documents[0]
            print(f"Page 1 Content Preview:")
            print("-"*50)
            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            print(content_preview)
            print("-"*50)
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print("="*50)

# Example usage
if __name__ == "__main__":
    loader = DocumentLoader()
    documents = loader.load_pdf(os.path.join("..", "data", "stats.pdf"), verbose=True)
    split_docs = loader.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks")