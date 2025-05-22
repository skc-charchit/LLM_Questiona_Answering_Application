# Document Question Answering Application

## Problem Statement

Organizations and individuals often need to quickly extract information from large documents without reading through the entire content. Traditional search methods may not understand the context or provide precise answers to specific questions. This application addresses this challenge by allowing users to:

1. Upload various document types (PDF, DOCX, TXT, etc.)
2. Ask natural language questions about the document content
3. Receive accurate, contextual answers based on the document's information

The application leverages advanced AI to understand document content and provide relevant answers, saving time and improving information retrieval efficiency.

## Technologies Used

### Core Technologies
- **Python 3.9+**: Core programming language
- **Streamlit**: Web interface framework
- **LangChain**: Framework for building LLM applications
- **Together.ai**: Provider for the DeepSeek-V3 language model
- **FAISS**: Vector database for efficient similarity search

### Key Libraries
- **langchain**: Core LLM application framework
- **langchain-community**: Community integrations for document loaders
- **langchain-text-splitters**: Text chunking functionality
- **together**: Together.ai Python SDK
- **streamlit**: UI dashboards and chat interfaces
- **requests**: HTTP requests for API calls
- **python-dotenv**: Environment variable management

### Document Processing
- **pypdf**: PDF parsing
- **unstructured**: Parsing unstructured data
- **docx2txt**: Word document parsing
- **openpyxl**: Excel file parsing
- **python-pptx**: PowerPoint file parsing
- **beautifulsoup4**: HTML parsing

## Project Structure

```
LLM_Questiona_Answering_Application/
├── .env                    # Environment variables and API keys
├── main.py                 # Application entry point
├── requirements.txt        # Project dependencies
├── src/                    # Source code directory
│   ├── app.py              # Main Streamlit application
│   ├── data/               # Sample data directory
│   │   └── stats.pdf       # Example PDF for testing
│   ├── pipelines/          # LLM pipelines
│   │   └── qa_chain.py     # Question answering chain
│   └── tasks/              # Core functionality modules
│       ├── embeddings.py   # Document embedding functionality
│       ├── loader.py       # Document loading and processing
│       └── together_ai.py  # Together.ai integration
```

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Together.ai API key (sign up at [together.ai](https://together.ai))

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LLM_Questiona_Answering_Application.git
   cd LLM_Questiona_Answering_Application
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your API key:
   - Create a `.env` file in the root directory
   - Add your Together.ai API key:
     ```
     TOGETHER_API_KEY=your_together_api_key_here
     TOGETHER_MODEL_NAME=deepseek-ai/DeepSeek-V3
     TOGETHER_EMBEDDING_MODEL=togethercomputer/m2-bert-80M-8k-retrieval
     ```

## How to Run the Application

1. Start the Streamlit application:
   ```bash
   streamlit run main.py
   ```

2. The application will open in your default web browser (typically at http://localhost:8501)

3. Using the application:
   - Upload a document using the "Upload Document" tab
   - Alternatively, enter a URL in the "From URL" tab or paste text in the "From Text" tab
   - Once the document is processed, ask questions in the chat input at the bottom
   - View the AI's responses in the chat interface

## Application Workflow

1. **Document Processing**:
   - `src/app.py` handles document uploads through the Streamlit interface
   - `src/tasks/loader.py` loads and processes documents based on file type
   - Documents are split into chunks using `RecursiveCharacterTextSplitter`

2. **Embedding and Indexing**:
   - `src/tasks/embeddings.py` converts document chunks into vector embeddings
   - Embeddings are stored in a FAISS vector database for efficient retrieval

3. **Question Answering**:
   - User questions are processed by `src/pipelines/qa_chain.py`
   - The question is converted to an embedding and used to retrieve relevant document chunks
   - Together.ai's DeepSeek-V3 model generates an answer based on the retrieved context
   - The answer is displayed to the user in the chat interface

## Output Screenshots

![Document Processing](https://drive.google.com/file/d/1WNeuM4O6JtxDBI3JMMClpChSlOUrDStx/view?usp=sharing)
*Document processing and content preview*

## Limitations & Future Improvements

### Current Limitations
- Limited to text-based content in documents
- May struggle with complex tables or highly specialized content
- Performance depends on the quality of the Together.ai API connection
- No persistent storage of document embeddings between sessions

### Future Improvements
- Add support for image-based document understanding
- Implement persistent storage for document embeddings
- Add multi-user support with authentication
- Improve handling of tables and structured data
- Add document comparison functionality
- Implement citation of sources in responses
- Add support for more languages
- Create a mobile-friendly interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Together.ai for providing the DeepSeek-V3 model
- LangChain for the excellent framework
- Streamlit for the intuitive UI components
