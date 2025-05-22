import os
import streamlit as st
from dotenv import load_dotenv
from src.tasks.loader import DocumentLoader
from src.tasks.embeddings import EmbeddingManager
from src.pipelines.qa_chain import QAChain

# Load environment variables
load_dotenv()

class PDFQAApp:
    """Streamlit application for PDF question answering."""

    def __init__(self):
        """Initialize the application."""
        self.doc_loader = DocumentLoader()
        self.embedding_manager = EmbeddingManager()
        self.qa_chain = QAChain()

        # Set up session state
        if "conversation" not in st.session_state:
            st.session_state.conversation = None

    def setup_ui(self):
        """Set up the user interface."""
        st.set_page_config(page_title="Chatbot for Q&A", page_icon="💬")
        st.title("💬 Chatbot for Question Answering")

        # Sidebar info
        with st.sidebar:
            st.title("📚 LLM ChatApp using Langchain")
            st.markdown("""
                This is an LLM-powered chatbot for answering questions from PDF documents.
            """)

    def handle_pdf_upload(self):
        """Handle PDF upload and processing."""
        pdf_file = st.file_uploader("📄 Upload a PDF document", type="pdf")

        if pdf_file is not None:
            with st.spinner("Processing PDF..."):
                # Save the uploaded file to a temp location
                temp_pdf_path = os.path.join("temp_uploaded.pdf")
                with open(temp_pdf_path, "wb") as f:
                    f.write(pdf_file.getbuffer())

                # Load and process the PDF
                documents = self.doc_loader.load_pdf(temp_pdf_path)

                # Display PDF info
                if documents:
                    doc = documents[0]
                    st.write("**PDF Loaded Successfully!**")
                    st.write(f"Total pages: {len(documents)}")
                    content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    st.write("**Page 1 Preview:**")
                    st.write(content_preview)

                # Split documents and create vector store
                split_docs = self.doc_loader.split_documents(documents)
                vectorstore = self.embedding_manager.create_vector_store(split_docs)

                # Create conversation chain
                conversation_chain = self.qa_chain.create_chain(vectorstore)
                st.session_state.conversation = conversation_chain

                st.success("PDF uploaded and processed successfully!")

    def handle_user_input(self):
        """Handle user input and generate responses."""
        if st.session_state.conversation:
            user_question = st.text_input("💬 Ask a question about your PDF:")
            if user_question:
                with st.spinner("Generating response..."):
                    response = st.session_state.conversation({"question": user_question})
                    st.write("**Answer:**")
                    st.write(response["answer"])

    def run(self):
        """Run the application."""
        self.setup_ui()
        self.handle_pdf_upload()
        self.handle_user_input()

# Example usage
if __name__ == "__main__":
    app = PDFQAApp()
    app.run()
