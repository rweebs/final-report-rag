# Rahmat Wibowo's Undergraduate Final Report QA Chatbot

## Purpose
This project is an interactive question-answering (QA) chatbot that allows users to ask questions about Rahmat Wibowo's undergraduate final report (TA2). The system loads the PDF, splits it into chunks, generates embeddings, stores them in a vector database, and uses retrieval-augmented generation (RAG) to answer user queries with context from the document.

## Tech Stack
- **Python 3.10+**
- **Streamlit**: For the web-based user interface
- **ChromaDB**: Vector database for storing and retrieving document embeddings
- **Ollama**: Local LLM server with OpenAI-compatible API (for both chat and embedding models)
- **openai** Python package: For interacting with Ollama's OpenAI-compatible API
- **PyPDF2**: For extracting text from PDF files
- **python-dotenv**: For environment variable management

## Setup Instructions

### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd rag-intro-chat-with-docs
```

### 2. Install Python Dependencies
It is recommended to use a virtual environment:
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install and Set Up Ollama
- Download and install Ollama from [https://ollama.com/](https://ollama.com/)
- Start the Ollama server:
  ```sh
  ollama serve
  ```
- Pull the required models:
  - For embeddings:
    ```sh
    ollama pull nomic-embed-text
    ```
  - For chat (LLM):
    ```sh
    ollama pull llama3
    ```

### 4. Prepare the Data
- Place your PDF file in the `data/` directory. By default, the app uses `data/TA2_English_Rahmat Wibowo.pdf`.

### 5. Set Up Environment Variables (Optional)
- Create a `.env` file if you need to override any environment variables. By default, Ollama does not require an API key.

## How to Run the Service

Start the Streamlit app:
```sh
streamlit run app.py
```

This will launch a web interface where you can enter your questions about the final report and receive concise, context-aware answers.

## Notes
- Make sure Ollama is running and the required models are pulled before starting the app.
- The first run may take longer as embeddings are generated and stored in ChromaDB.
- You can change the embedding or chat model by editing the `MODEL_EMBEDDING` or chat model name in `app.py`.
