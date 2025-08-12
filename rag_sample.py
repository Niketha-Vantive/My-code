# step5_rag_app.py
# Implements ONLY Step 5 from your doc:
#  - Step 5.1: query embedding + hybrid search (semantic + vector)
#  - Step 5.2: pass concatenated chunk context to AOAI chat for the answer
#
# Uses env vars from your .env (no hardcoded secrets).

import os
import streamlit as st
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

load_dotenv()  # read .env

# ─────────────────────────────
# Env config required for Step 5
# ─────────────────────────────
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")           # https://<search>.search.windows.net
AZURE_SEARCH_INDEX    = os.getenv("AZURE_SEARCH_INDEX")              # your index name
AZURE_SEARCH_API_KEY  = os.getenv("AZURE_SEARCH_API_KEY")            # admin/query key
SEMANTIC_CONFIG_NAME  = os.getenv("SEMANTIC_CONFIG_NAME", "default")

# Index field names from your "Import & Vectorize" wizard
CHUNK_TEXT_FIELD      = os.getenv("CHUNK_TEXT_FIELD", "chunk")       # text field (e.g., "chunk" or "content")
CHUNK_VECTOR_FIELD    = os.getenv("CHUNK_VECTOR_FIELD", "text_vector")  # vector field

# Azure OpenAI (embedding + chat)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")           # https://<aoai>.openai.azure.com/
AZURE_OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VER  = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
EMBED_DEPLOYMENT      = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")  # e.g., "text-embedding-3-small"
CHAT_DEPLOYMENT       = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")       # e.g., "gpt-4o"

DEFAULT_QUERY         = os.getenv("QUERY_DEFAULT", "What is the capital of France?")

# ─────────────────────────────
# Clients
# ─────────────────────────────
# Azure AI Search
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)

# Azure OpenAI (one client for embeddings + chat)
aoai = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VER,
)

# ─────────────────────────────
# Step 5.1 — embed + hybrid search
# ─────────────────────────────
def get_embedding(text: str) -> list[float]:
    resp = aoai.embeddings.create(model=EMBED_DEPLOYMENT, input=text)
    return resp.data[0].embedding

def search_documents(user_query: str, k: int = 10):
    # Build query embedding
    query_vector = get_embedding(user_query)

    # Hybrid retrieval: semantic + vector
    results_iter = search_client.search(
        search_text=user_query,                         # semantic boost
        query_type="semantic",
        semantic_configuration_name=SEMANTIC_CONFIG_NAME,
        vector_queries=[{
            "vector": query_vector,
            "k": k,
            "fields": CHUNK_VECTOR_FIELD,
            "kind": "vector",
        }],
        select="*",
        include_total_count=True,
    )

    docs = list(results_iter)
    # Concatenate chunk text for Step 5.2
    context = "\n".join([doc.get(CHUNK_TEXT_FIELD, "[no chunk]") for doc in docs])
    return docs, context

# ─────────────────────────────
# Step 5.2 — generate answer with chat
# ─────────────────────────────
def generate_answer(context: str, user_query: str) -> str:
    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question:
{user_query}
"""
    resp = aoai.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip()

# ─────────────────────────────
# Minimal UI for Step 5
# ─────────────────────────────
st.set_page_config(page_title="Step 5 RAG — Azure AI Search + AOAI", layout="wide")
st.title("Step 5: RAG (Hybrid Search + Chat)")

query_text = st.text_input("Your question", value=DEFAULT_QUERY)
k = st.slider("Top K chunks", 3, 20, 10, 1)

if query_text.strip():
    try:
        docs, context = search_documents(query_text, k=k)

        st.subheader("Top hits (first 5)")
        st.json(docs[:min(5, len(docs))])

        st.subheader(f"Concatenated context from '{CHUNK_TEXT_FIELD}'")
        st.code(context[:3000] + ("..." if len(context) > 3000 else ""))

        st.subheader("Answer")
        if context.strip():
            answer = generate_answer(context, query_text)
            st.write(answer)
        else:
            st.warning(
                f"No text found in field '{CHUNK_TEXT_FIELD}'. "
                "Check your index field names and that your index has data."
            )

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Verify .env values (endpoints, keys, deployment names) and index field names.")
else:
    st.info("Enter a question to run Step 5.")

 
