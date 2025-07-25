import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
import PyPDF2

# Load environment variables from .env file
load_dotenv()


# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=None
)


client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama', # Placeholder, as Ollama doesn't require a real API key
)


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


def load_text_from_pdf(pdf_path):
    print("==== Loading text from PDF ====")
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Load text from the PDF file
pdf_path = "data/TA2_English_Rahmat Wibowo.pdf"
pdf_text = load_text_from_pdf(pdf_path)

documents = [{"id": "TA2_English_Rahmat_Wibowo", "text": pdf_text}]

print(f"Loaded {len(documents)} document from PDF")

# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting PDF into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# print(f"Split documents into {len(chunked_documents)} chunks")


# Model constants
MODEL_EMBEDDING = "nomic-embed-text"  # Change if you use a different embedding model

# Function to generate embeddings using OpenAI API
def get_openai_embedding(text):
    try:
        response = client.embeddings.create(input=text, model=MODEL_EMBEDDING)
        embedding = response.data[0].embedding
        print("==== Generating embeddings... ====")
        return embedding
    except Exception as e:
        print(f"\n[ERROR] Failed to generate embedding with model '{MODEL_EMBEDDING}'.\nReason: {e}\n")
        print(f"If you are using Ollama, make sure to pull the model first:\n  ollama pull {MODEL_EMBEDDING}\n")
        raise


# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_openai_embedding(doc["text"])

# print(doc["embedding"])

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )


# Function to query documents
def query_documents(question, n_results=2):
    query_embedding = get_openai_embedding(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks
    # for idx, document in enumerate(results["documents"][0]):
    #     doc_id = results["ids"][0][idx]
    #     distance = results["distances"][0][idx]
    #     print(f"Found document chunk: {document} (ID: {doc_id}, Distance: {distance})")


# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message.content
    return answer


# Example query
# query_documents("tell me about AI replacing TV writers strike.")
# Example query and response generation
# question = "Who is Rahmat WIbowo?"
# relevant_chunks = query_documents(question)
# answer = generate_response(question, relevant_chunks)

# print(answer)

import streamlit as st
st.title("Rahmat Wibowo's Undergraduate Final Report")
question = st.text_area("Describe what you want to ask to Rahmat Wibowo")
relevant_chunks = query_documents(question)
if st.button("Ask"):
    answer = generate_response(question, relevant_chunks)
    st.write(answer)
