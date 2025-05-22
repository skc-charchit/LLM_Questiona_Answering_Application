import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class QAChain:
    """Class for creating and managing question-answering chains."""

    def __init__(self, model_repo_id=None, temperature=0.01, max_length=512, memory_k=5):
        """
        Initialize with model parameters.

        Args:
            model_repo_id: Hugging Face model repository ID
            temperature: Temperature for text generation
            max_length: Maximum length of generated text
            memory_k: Number of conversation turns to keep in memory
        """
        self.model_repo_id = model_repo_id or os.getenv(
            "LLM_MODEL_REPO_ID",
            "TheBloke/phi-2-GGUF"     
            )
        
        self.temperature = temperature
        self.max_length = max_length
        self.memory_k = memory_k

        # Initialize LLM
        self.llm = HuggingFaceHub(
            repo_id=self.model_repo_id,
            model_kwargs={"temperature": self.temperature, "max_length": self.max_length}
        )

        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=self.memory_k
        )

        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="You are an expert assistant. Use the following PDF content to answer the user's question.\n\nContent: {context}\n\nQuestion: {question}\nAnswer:"
        )

    def create_chain(self, vectorstore):
        """
        Create a conversational retrieval chain.

        Args:
            vectorstore: Vector store to use for retrieval

        Returns:
            ConversationalRetrievalChain
        """
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt_template}
        )

        return conversation_chain

    def get_answer(self, chain, question):
        """
        Get an answer to a question.

        Args:
            chain: ConversationalRetrievalChain
            question: Question to answer

        Returns:
            Answer to the question
        """
        response = chain({"question": question})
        return response["answer"]

# Example usage
if __name__ == "__main__":
    import sys
    import os

    # Add parent directory to path to import from sibling packages
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from src.tasks.loader import DocumentLoader
    from src.tasks.embeddings import EmbeddingManager

    # Load and process documents
    doc_loader = DocumentLoader()
    documents = doc_loader.load_pdf(os.path.join("..", "data", "stats.pdf"), verbose=True)
    split_docs = doc_loader.split_documents(documents)

    # Create vector store
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.create_vector_store(split_docs)

    # Create QA chain
    qa_chain = QAChain()
    chain = qa_chain.create_chain(vectorstore)

    # Ask a question
    question = "What is the main topic of this document?"
    answer = qa_chain.get_answer(chain, question)

    print("\n" + "="*50)
    print("QUESTION & ANSWER")
    print("="*50)
    print(f"Question: {question}")
    print("-"*50)
    print(f"Answer: {answer}")
    print("="*50)
