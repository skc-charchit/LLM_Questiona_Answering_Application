import os
import streamlit as st
from dotenv import load_dotenv
from src.tasks.loader import DocumentLoader
from src.tasks.embeddings import EmbeddingManager
from src.tasks.together_ai import TogetherAIManager
from src.pipelines.qa_chain import QAChain

# Load environment variables
load_dotenv()

class PDFQAApp:
    """Streamlit application for PDF question answering."""

    def __init__(self):
        """Initialize the application."""
        try:
            # Initialize Together.ai for LLM and embeddings
            self.together_ai = TogetherAIManager()

            # Get Together.ai LLM
            llm = self.together_ai.get_llm()

            # Initialize components
            self.doc_loader = DocumentLoader()
            self.embedding_manager = EmbeddingManager()
            self.qa_chain = QAChain(llm=llm)

            # Set up session state
            if "conversation" not in st.session_state:
                st.session_state.conversation = None

            self.error = None

        except Exception as e:
            self.error = str(e)
            print(f"Error initializing application: {e}")

    def setup_ui(self):
        """Set up the user interface."""
        st.set_page_config(page_title="Chatbot for Q&A", page_icon="💬")
        st.title("💬 Chatbot for Question Answering with Together.ai")

        # Display any initialization errors
        if hasattr(self, 'error') and self.error:
            st.error(f"Error initializing application: {self.error}")
            st.warning("Please check your API keys in the .env file.")
            return False

        # Sidebar info
        with st.sidebar:
            st.title("📚 LLM ChatApp using Together.ai")
            st.markdown("""
                This is a Together.ai-powered chatbot for answering questions from documents.

                **Features:**
                - Uses Together.ai's DeepSeek-V3 model for question answering
                - Uses Together.ai's embedding model for document retrieval
                - Processes various document types (PDF, DOCX, TXT, etc.)
                - Provides conversational responses to your questions
            """)

            # Display model information
            st.subheader("Model Information")

            # LLM info
            if hasattr(self, 'together_ai'):
                st.info(f"LLM: {self.together_ai.model_name} (Together.ai)")
            else:
                st.info(f"LLM: {os.getenv('TOGETHER_MODEL_NAME')} (Together.ai)")

            # Embedding info
            if hasattr(self, 'embedding_manager') and hasattr(self.embedding_manager, 'model_name'):
                st.info(f"Embeddings: {self.embedding_manager.model_name} (Together.ai)")
            else:
                st.info(f"Embeddings: {os.getenv('TOGETHER_EMBEDDING_MODEL')} (Together.ai)")

        return True

    def handle_document_upload(self):
        """Handle document upload and processing."""
        # Create tabs for different input methods
        upload_tab, url_tab, text_tab = st.tabs(["Upload Document", "From URL", "From Text"])

        with upload_tab:
            # Allow multiple file types
            uploaded_file = st.file_uploader(
                "📄 Upload a document",
                type=["pdf", "txt", "docx", "csv", "xlsx", "pptx", "html"]
            )

            if uploaded_file is not None:
                with st.spinner("Processing document..."):
                    # Get file extension
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

                    # Save the uploaded file to a temp location
                    temp_file_path = os.path.join(f"temp_uploaded{file_extension}")
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load and process the document
                    documents = self.doc_loader.load_document(temp_file_path)

                    # Process the documents
                    self._process_documents(documents, f"Document '{uploaded_file.name}'")

        with url_tab:
            url = st.text_input("🔗 Enter a URL to a document or webpage:")
            process_url = st.button("Process URL")

            if url and process_url:
                with st.spinner("Loading content from URL..."):
                    try:
                        # Load document from URL
                        documents = self.doc_loader.load_from_url(url, verbose=True)

                        # Process the documents
                        self._process_documents(documents, f"Content from URL: {url}")
                    except Exception as e:
                        st.error(f"Error loading URL: {str(e)}")

        with text_tab:
            text = st.text_area("✏️ Enter text directly:", height=200)
            process_text = st.button("Process Text")

            if text and process_text:
                with st.spinner("Processing text..."):
                    # Create document from text
                    documents = self.doc_loader.load_from_text(text)

                    # Process the documents
                    self._process_documents(documents, "User-provided text")

    def _process_documents(self, documents, source_description):
        """Process documents and create conversation chain."""
        if not documents:
            st.error("No content found in the document.")
            return

        # Display document info
        doc = documents[0]
        st.write(f"**{source_description} Loaded Successfully!**")
        st.write(f"Total sections: {len(documents)}")
        content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        st.write("**Content Preview:**")
        st.write(content_preview)

        # Split documents and create vector store
        split_docs = self.doc_loader.split_documents(documents)
        vectorstore = self.embedding_manager.create_vector_store(split_docs)

        # Store in session state
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = vectorstore
        else:
            st.session_state.vectorstore = vectorstore

        # Create conversation chain
        conversation_chain = self.qa_chain.create_chain(vectorstore)
        st.session_state.conversation = conversation_chain

        st.success(f"{source_description} processed successfully!")

    def handle_user_input(self):
        """Handle user input and generate responses."""
        if st.session_state.conversation:
            # Initialize chat history if it doesn't exist
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Display chat history
            for question, answer in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)

            # Get user input
            user_question = st.chat_input("💬 Ask a question about your document:")

            if user_question:
                # Display user question
                with st.chat_message("user"):
                    st.write(user_question)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.conversation({"question": user_question})
                            answer = response.get("answer", "I couldn't find an answer to that question in the document.")

                            # Display answer
                            st.write(answer)

                            # Add to chat history
                            st.session_state.chat_history.append((user_question, answer))
                        except Exception as e:
                            error_message = str(e)
                            if "429" in error_message or "quota" in error_message.lower():
                                error_text = "⚠️ API rate limit exceeded. Please try again later or reduce your usage."
                                st.error(error_text)
                                # Add to chat history
                                st.session_state.chat_history.append((user_question, error_text))
                                print(f"Rate limit error: {e}")
                            else:
                                error_text = f"Error generating response: {error_message}"
                                st.error(error_text)
                                # Add to chat history
                                st.session_state.chat_history.append((user_question, error_text))
                                print(f"Response generation error details: {e}")

            # Add option to clear chat history
            if st.session_state.chat_history and st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.experimental_rerun()

    def run(self):
        """Run the application."""
        # Setup UI and check for initialization errors
        if self.setup_ui():
            # Add app description
            st.markdown("""
            This application allows you to upload documents and ask questions about them.
            The app uses Together.ai's DeepSeek-V3 model to provide accurate answers based on the document content.

            **Supported document types:**
            - PDF files (.pdf)
            - Text files (.txt)
            - Word documents (.docx)
            - Excel spreadsheets (.xlsx)
            - PowerPoint presentations (.pptx)
            - CSV files (.csv)
            - HTML files (.html)
            - Web pages (via URL)
            - Direct text input
            """)

            # Handle document upload and user input
            self.handle_document_upload()
            self.handle_user_input()

            # Add footer
            st.markdown("---")
            st.markdown("Powered by Together.ai and LangChain")
        else:
            # Display instructions for fixing API key issues
            st.markdown("""
            ### Troubleshooting

            To fix API key issues:

            1. Make sure you have your Together.ai API key in the `.env` file:
            ```
            TOGETHER_API_KEY=your_together_api_key_here
            ```

            2. Get an API key from [Together.ai](https://together.ai/)

            3. Restart the application after updating the key
            """)

# Example usage
if __name__ == "__main__":
    app = PDFQAApp()
    app.run()
