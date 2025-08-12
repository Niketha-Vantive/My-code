import os

import glob

from docx import Document

import chromadb

# from openai import OpenAI

import time # For exponential backoff

import os  

import base64

from openai import AzureOpenAI  

EMBEDDING_MODEL = "text-embedding-3-small"

deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  

# --- Configuration ---

# Set your OpenAI API key as an environment variable (recommended)

# export OPENAI_API_KEY="your_api_key_here"

# Alternatively, you can hardcode it here, but it's less secure:

# OPENAI_API_KEY = "your_openai_api_key"

DOCS_DIRECTORY = "rag_res" # Directory where your .docx files are located

CHROMA_DB_PATH = "chroma_db"     # Path to store your ChromaDB data

COLLECTION_NAME = "docx_knowledge_base"

CHUNK_SIZE = 1000  # Max characters per text chunk

CHUNK_OVERLAP = 100 # Overlap between chunks to maintain context


def load_docx_documents(directory):

    """

    Loads all text from .docx files in the specified directory.

    Returns a list of dictionaries, each containing 'text' and 'source'.

    """

    documents = []

    print(f"Loading documents from: {directory}")

    docx_files = glob.glob(os.path.join(directory, "*.docx"))

    if not docx_files:

        print(f"No .docx files found in '{directory}'. Please ensure your reference documents are in this directory.")

        return []

    for filepath in docx_files:

        try:

            doc = Document(filepath)

            full_text = []

            for para in doc.paragraphs:

                full_text.append(para.text)

            text_content = "\n".join(full_text)

            documents.append({"text": text_content, "source": os.path.basename(filepath)})

            print(f"Loaded: {os.path.basename(filepath)}")

        except Exception as e:

            print(f"Error loading {filepath}: {e}")

    return documents

def chunk_text(text, chunk_size, overlap):

    """

    Splits a long text into smaller chunks with optional overlap.

    """

    chunks = []

    if not text:

        return chunks

    current_position = 0

    while current_position < len(text):

        end_position = min(current_position + chunk_size, len(text))

        chunk = text[current_position:end_position]

        chunks.append(chunk)

        current_position += chunk_size - overlap

        if current_position >= len(text) and end_position < len(text):

            # Ensure the last bit is not missed if it's smaller than chunk_size - overlap

            chunks.append(text[end_position:])

            break

    return chunks

def get_embeddings(client, texts, retries=5, delay=1):

    """

    Generates embeddings for a list of texts using OpenAI's embedding model.

    Implements exponential backoff for API calls.

    """

    embeddings = []

    for i, text in enumerate(texts):

        for attempt in range(retries):

            try:

                response = client.embeddings.create(

                    input=[text],

                    model=EMBEDDING_MODEL

                )

                embeddings.append(response.data[0].embedding)

                break # Success, break out of retry loop

            except Exception as e:

                print(f"Error getting embedding for chunk {i+1} (Attempt {attempt + 1}/{retries}): {e}")

                if attempt < retries - 1:

                    time.sleep(delay * (2 ** attempt)) # Exponential backoff

                else:

                    print(f"Failed to get embedding for text after {retries} attempts: {text[:50]}...")

                    embeddings.append(None) # Append None or handle as appropriate

    return embeddings

def setup_chroma_db(client, documents_dir, chroma_db_path, collection_name):

    """

    Loads documents, chunks them, generates embeddings, and populates ChromaDB.

    Returns the ChromaDB collection.

    """

    print("\n--- Setting up knowledge base ---")

    client_chroma = chromadb.PersistentClient(path=chroma_db_path)

    # Try to get the collection; create if it doesn't exist

    try:

        collection = client_chroma.get_collection(name=collection_name)

        print(f"Existing ChromaDB collection '{collection_name}' loaded.")

        # Optionally, you might want to clear it and re-add if documents change frequently

        # collection.delete() # Uncomment to clear and re-add all documents every run

        # collection = client_chroma.create_collection(name=collection_name)

        # print(f"Re-created ChromaDB collection '{collection_name}'.")

    except:

        collection = client_chroma.create_collection(name=collection_name)

        print(f"New ChromaDB collection '{collection_name}' created.")

    # Check if the collection is empty or if we need to re-populate

    if collection.count() == 0:

        print("Populating ChromaDB with document chunks...")

        documents = load_docx_documents(documents_dir)

        if not documents:

            print("No documents to process. Exiting setup.")

            return None

        all_chunks = []

        all_metadatas = []

        all_ids = []

        doc_id_counter = 0

        for doc in documents:

            chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)

            for i, chunk in enumerate(chunks):

                all_chunks.append(chunk)

                all_metadatas.append({"source": doc["source"], "chunk_index": i})

                all_ids.append(f"doc_{doc_id_counter}_chunk_{i}")

            doc_id_counter += 1

        print(f"Generated {len(all_chunks)} text chunks.")

        print("Generating embeddings for chunks (this may take a while)...")

        chunk_embeddings = get_embeddings(client, all_chunks)

        # Filter out None embeddings in case of API failures

        valid_chunks = []

        valid_embeddings = []

        valid_metadatas = []

        valid_ids = []

        for i, emb in enumerate(chunk_embeddings):

            if emb is not None:

                valid_embeddings.append(emb)

                valid_chunks.append(all_chunks[i])

                valid_metadatas.append(all_metadatas[i])

                valid_ids.append(all_ids[i])

        if valid_embeddings:

            collection.add(

                embeddings=valid_embeddings,

                documents=valid_chunks,

                metadatas=valid_metadatas,

                ids=valid_ids

            )

            print(f"Added {len(valid_embeddings)} chunks to ChromaDB.")

        else:

            print("No valid embeddings were generated. ChromaDB collection remains empty.")

    else:

        print(f"ChromaDB collection '{collection_name}' already contains {collection.count()} items. Skipping re-population.")

    return collection

def retrieve_context(client, query_text, collection, n_results=3):
    """
    Retrieves the most relevant document chunks from ChromaDB based on the query.
    """
    query_embedding = get_embeddings(client, [query_text])
    if not query_embedding or query_embedding[0] is None:
        print("Could not generate embedding for query. Cannot retrieve context.")
        return []
    results = collection.query(
        query_embeddings=query_embedding[0],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    return results['documents'][0] if results['documents'] else []
def generate_response(client, query, context_chunks):
    """
    Generates a response using GPT-3.5 Turbo, incorporating retrieved context.
    """
    context_str = "\n".join(context_chunks)
    if context_str:
        prompt = (
            f"Based on the following context, answer the question. "
            f"If the answer is not in the context, return None\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
    else:
        prompt = (
            f"Answer the following question. If you don't know, return None.\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
    messages = [
        {"role": "user", "content": prompt}
    ]
    messages = [
    {"role": "system", "content": "You are an assistant."},
    {"role": "user", "content": prompt}
    ]
    for attempt in range(5): # 5 retries for GPT API call
        try:
            # Generate the completion  
            completion = client.chat.completions.create(  
                model=deployment,
                messages=messages,
                stop=None,  
                stream=False
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response (Attempt {attempt + 1}/5): {e}")
            if attempt < 4:
                time.sleep(2 * (2 ** attempt)) # Exponential backoff
            else:
                return "I apologize, but I'm having trouble generating a response at the moment. Please try again later."

# --- Main Chatbot Logic ---
content_rag = {"chroma_collection" : None}
def initial_rag_setup(client):
    # Create the directory for reference documents if it doesn't exist
    if not os.path.exists(DOCS_DIRECTORY):
        os.makedirs(DOCS_DIRECTORY)
        print(f"Created directory: '{DOCS_DIRECTORY}'. Please place your .docx files here.")
        print("Exiting. Run the script again after adding documents.")
        return
    # Setup ChromaDB
    chroma_collection = setup_chroma_db(client, DOCS_DIRECTORY, CHROMA_DB_PATH, COLLECTION_NAME)
    if chroma_collection is None:
        print("Failed to set up ChromaDB. Please check the document directory and your OpenAI API key.")
        return
    print("\n--- Chatbot Ready ---")
    # print("Type 'exit' or 'quit' to end the conversation.")
    content_rag["chroma_collection"] = chroma_collection
def ask_question(client, user_query):
    print("Searching for relevant information...")
    context_chunks = retrieve_context(client, user_query, content_rag["chroma_collection"])
    if not context_chunks:
        print("No relevant context found in documents.")
        response = generate_response(client, user_query, []) # Try to answer without context
    else:
        print(f"Found {len(context_chunks)} relevant chunks. Generating response...")
        response = generate_response(client, user_query, context_chunks)
    print(f"\nChatbot: {response}")
    return response

# def main():
#     while True:
#         user_query = input("\nYour question: ")
#         if user_query.lower() in ["exit", "quit"]:
#             print("Goodbye!")
#             break
#         print("Searching for relevant information...")
#         context_chunks = retrieve_context(user_query, chroma_collection)
#         if not context_chunks:
#             print("No relevant context found in documents.")
#             response = generate_response(user_query, []) # Try to answer without context
#         else:
#             print(f"Found {len(context_chunks)} relevant chunks. Generating response...")
#             response = generate_response(user_query, context_chunks)
#         print(f"\nChatbot: {response}")
# if __name__ == "__main__":
    # main()

    # Explain change analysis design input CA1
