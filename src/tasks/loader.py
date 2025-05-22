import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    OnlinePDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

class DocumentLoader:
    """Class for loading and processing various document types."""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """Initialize with text splitting parameters."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap

        )

    def load_document(self, file_path, file_type=None, verbose=False):
        """
        Load a document from a path.

        Args:
            file_path: Path to the document file
            file_type: Type of the document (pdf, txt, docx, csv, xlsx, pptx, html)
                       If None, will be inferred from file extension
            verbose: Whether to print information about the loaded document

        Returns:
            List of Document objects
        """
        # Convert to absolute path if relative
        if not os.path.isabs(file_path):
            base_dir = Path(__file__).parent.parent.parent  # Go up to project root
            file_path = os.path.join(base_dir, file_path)

        # Infer file type from extension if not provided
        if file_type is None:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.pdf':
                file_type = 'pdf'
            elif file_extension in ['.txt', '.text', '.md', '.markdown']:
                file_type = 'txt'
            elif file_extension in ['.docx', '.doc']:
                file_type = 'docx'
            elif file_extension == '.csv':
                file_type = 'csv'
            elif file_extension in ['.xlsx', '.xls']:
                file_type = 'xlsx'
            elif file_extension in ['.pptx', '.ppt']:
                file_type = 'pptx'
            elif file_extension in ['.html', '.htm']:
                file_type = 'html'
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")

        # Load document based on file type
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_type == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_type == 'docx':
            loader = Docx2txtLoader(file_path)
        elif file_type == 'csv':
            loader = CSVLoader(file_path)
        elif file_type == 'xlsx':
            loader = UnstructuredExcelLoader(file_path)
        elif file_type == 'pptx':
            loader = UnstructuredPowerPointLoader(file_path)
        elif file_type == 'html':
            loader = UnstructuredHTMLLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Load documents
        documents = loader.load()

        if verbose and documents:
            self._print_document_info(documents)

        return documents

    def load_pdf(self, pdf_path, verbose=False):
        """
        Load a PDF file from a path.

        Args:
            pdf_path: Path to the PDF file
            verbose: Whether to print information about the loaded PDF

        Returns:
            List of Document objects
        """
        return self.load_document(pdf_path, file_type='pdf', verbose=verbose)

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

    def load_from_text(self, text, metadata=None, verbose=False):
        """
        Create a document from text.

        Args:
            text: Text content
            metadata: Optional metadata dictionary
            verbose: Whether to print information about the created document

        Returns:
            List containing a single Document object
        """
        if metadata is None:
            metadata = {"source": "user_input"}

        document = Document(page_content=text, metadata=metadata)
        documents = [document]

        if verbose:
            self._print_document_info(documents)

        return documents

    def load_from_url(self, url, verbose=False):
        """
        Load a document from a URL.

        Args:
            url: URL to load
            verbose: Whether to print information about the loaded document

        Returns:
            List of Document objects
        """
        # Check if it's a PDF URL
        if url.lower().endswith('.pdf'):
            return self.load_online_pdf(url, verbose=verbose)

        # For HTML and other web content, use UnstructuredHTMLLoader
        # Note: We're directly loading from URL, no need for a temporary file

        try:
            # Use UnstructuredHTMLLoader to load the URL
            loader = UnstructuredHTMLLoader(url)
            documents = loader.load()

            if verbose and documents:
                self._print_document_info(documents)

            return documents
        except Exception as e:
            print(f"Error loading URL: {e}")
            # Return an empty document with the URL as content
            return self.load_from_text(f"Failed to load content from {url}: {str(e)}")

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
        print("DOCUMENT INFORMATION")
        print("="*50)
        print(f"Total sections loaded: {len(documents)}")
        print("-"*50)

        if documents:
            doc = documents[0]
            print(f"First Section Content Preview:")
            print("-"*50)
            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            print(content_preview)
            print("-"*50)
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print("="*50)

