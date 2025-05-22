from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

class QAChain:
    """Class for creating and managing question-answering chains."""

    def __init__(self, llm=None, temperature=0.7, max_output_tokens=2048, memory_k=5):
        """
        Initialize with model parameters.

        Args:
            llm: Language model instance (if None, will be set by the caller)
            temperature: Temperature for text generation
            max_output_tokens: Maximum number of tokens to generate
            memory_k: Number of conversation turns to keep in memory
        """
        self.llm = llm  # Will be set by the caller if None
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.memory_k = memory_k

        if self.llm:
            print("Language model provided externally")
        else:
            print("Warning: No language model provided. Make sure to set it before creating a chain.")

        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=self.memory_k
        )

        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="You are an expert assistant. Use the following document content to answer the user's question.\n\nContent: {context}\n\nQuestion: {question}\nAnswer:"
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
            retriever=vectorstore.as_retriever(search_type="similarity"),
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
        response = chain.invoke({"question": question})
        return response.get("answer", "No answer found.")


