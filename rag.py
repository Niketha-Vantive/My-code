# app.py
# Streamlit RAG with Azure AI Search (hybrid) + Azure OpenAI (chat)
# Fill the TODOs below and run: streamlit run app.py

import os
import sys
import traceback
import streamlit as st
from typing import List, Tuple

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# =========================
# 🔧 CONFIG — FILL THESE
# =========================
# --- Azure AI Search ---
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://<YOUR-SEARCH-NAME>.search.windows.net")  # TODO
AZURE_SEARCH_INDEX    = os.getenv("AZURE_SEARCH_INDEX",    "<YOUR-INDEX-NAME>")                               # TODO
AZURE_SEARCH_API_KEY  = os.getenv("AZURE_SEARCH_API_KEY",  "<YOUR-SEARCH-ADMIN-KEY>")                         # TODO

# Field names created by the "Import and Vectorize" wizard (adjust if your index differs)
CHUNK_TEXT_FIELD      = os.getenv("CHUNK_TEXT_FIELD", "chunk")            # e.g., "chunk" or "content"
CHUNK_VECTOR_FIELD    = os.getenv("CHUNK_VECTOR_FIELD", "text_vector")    # e.g., "text_vector" or "contentVector"
SEMANTIC_CONFIG_NAME  = os.getenv("SEMANTIC_CONFIG_NAME", "default")      # name of your semantic configuration

# --- Azure OpenAI ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://<YOUR-AOAI-NAME>.openai.azure.com/")  # TODO
AZURE_OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY",  "<YOUR-AOAI-KEY>")                             # TODO

# Your *deployment names* (NOT model IDs)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")  # TODO
AZURE_OPENAI_CHAT_DEPLOYMENT      = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT",      "gpt-4o")                  # TODO

# API version (keep in sync with your AOAI resource)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# =========================
# 🔗 CLIENTS
# =========================
# Azure AI Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

# Azure OpenAI client
aoai = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

# =========================
# 🧠 EMBEDDINGS
# =========================
def get_embedding(text: str) -> List[float]:
    """Return embedding vector for the given text using your AOAI embedding deployment."""
    resp = aoai.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=text
    )
    return resp.data[0].embedding

# =========================
# 🔎 HYBRID SEARCH (semantic + vector)
# =========================
def search_documents(
    user_query: str,
    k: int = 8,
    chunk_field: str = CHUNK_TEXT_FIELD,
    vector_field: str = CHUNK_VECTOR_FIELD,
    semantic_config: str = SEMANTIC_CONFIG_NAME
) -> Tuple[list, str, object]:
    """
    Perform hybrid search:
    - semantic re-ranking via search_text
    - vector similarity via vector_queries
    Returns (docs_list, concatenated_context, raw_results_iterator)
    """
    query_vector = get_embedding(user_query)

    results = search_client.search(
        search_text=user_query,  # semantic boost
        query_type="semantic",
        semantic_configuration_name=semantic_config,
        vector_queries=[{
            "vector": query_vector,
            "k": k,
            "fields": vector_field,
            "kind": "vector"
        }],
        select="*",
        include_total_count=True,
        query_caption="extractive",
        query_caption_highlight_enabled=True
    )

    docs = list(results)
    context = "\n\n".join([doc.get(chunk_field, "[no chunk in this doc]") for doc in docs])
    return docs, context, results

# =========================
# ✍️ GENERATE ANSWER
# =========================
def generate_answer(context: str, user_query: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
    """Use AOAI Chat completion with retrieved context to answer user_query."""
    prompt = f"""You are a helpful assistant. Answer the question strictly using the given context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{user_query}
"""

    resp = aoai.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# =========================
# 🖥️ STREAMLIT UI
# =========================
st.set_page_config(page_title="Azure RAG (AI Search + AOAI)", layout="wide")
st.title("🔎📚 RAG with Azure AI Search + Azure OpenAI")

# Sidebar config
with st.sidebar:
    st.header("Settings")
    default_query = "What is the capital of France?"
    user_query = st.text_input("Your question", value=default_query)
    top_k = st.slider("Top K chunks", min_value=3, max_value=20, value=8, step=1)
    temperature = st.slider("Answer temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    max_tokens = st.slider("Max tokens", min_value=100, max_value=2000, value=700, step=50)

    st.subheader("Index field names")
    st.text_input("Chunk text field", key="chunk_field_name", value=CHUNK_TEXT_FIELD)
    st.text_input("Chunk vector field", key="chunk_vec_field_name", value=CHUNK_VECTOR_FIELD)
    st.text_input("Semantic config", key="semantic_config_name", value=SEMANTIC_CONFIG_NAME)

tabs = st.tabs(["🔍 Search Hits", "💡 @search.answers", "🧩 First Chunk", "🤖 Final Answer", "⚙️ Debug"])

if st.sidebar.button("Run RAG", type="primary"):
    try:
        # Use any overridden field names from sidebar
        chunk_field_ui = st.session_state.get("chunk_field_name", CHUNK_TEXT_FIELD)
        vector_field_ui = st.session_state.get("chunk_vec_field_name", CHUNK_VECTOR_FIELD)
        semantic_cfg_ui = st.session_state.get("semantic_config_name", SEMANTIC_CONFIG_NAME)

        # 1) Search
        docs, context, results = search_documents(
            user_query,
            k=top_k,
            chunk_field=chunk_field_ui,
            vector_field=vector_field_ui,
            semantic_config=semantic_cfg_ui
        )

        # Tab 1 — raw hits
        with tabs[0]:
            st.write(f"Total hits: {len(docs)}")
            st.json(docs[:min(5, len(docs))])

        # Tab 2 — @search.answers (SDK support varies)
        with tabs[1]:
            try:
                # Some versions of azure-search-documents expose this; if not, we show a friendly message.
                answers = results.get_answers()  # type: ignore[attr-defined]
                if answers:
                    st.write("First @search.answers:")
                    st.write(answers[0].text)
                else:
                    st.info("No @search.answers found.")
            except Exception:
                st.info("Your SDK version may not expose @search.answers. Upgrade azure-search-documents or ignore this tab.")

        # Tab 3 — first chunk
        with tabs[2]:
            if docs:
                first_chunk = docs[0].get(chunk_field_ui)
                if first_chunk:
                    st.write(first_chunk)
                else:
                    st.warning(f"No '{chunk_field_ui}' field on the first document. Check your index field mapping.")
            else:
                st.info("No documents returned.")

        # 2) Generate final answer
        with tabs[3]:
            if context.strip():
                answer = generate_answer(context, user_query, temperature=temperature, max_tokens=max_tokens)
                st.write(answer)
            else:
                st.warning("No context retrieved from Search. Check your index fields and that your index has data.")

        # Debug info
        with tabs[4]:
            st.code(
                f"Endpoint/Search Index: {AZURE_SEARCH_ENDPOINT} / {AZURE_SEARCH_INDEX}\n"
                f"Fields: text={chunk_field_ui}, vector={vector_field_ui}, semantic='{semantic_cfg_ui}'\n"
                f"AOAI Endpoint: {AZURE_OPENAI_ENDPOINT}\n"
                f"Embeddings: {AZURE_OPENAI_EMBEDDING_DEPLOYMENT} | Chat: {AZURE_OPENAI_CHAT_DEPLOYMENT}\n",
                language="text"
            )

    except Exception as e:
        with tabs[4]:
            st.error(f"Error: {e}")
            st.text("Traceback:")
            st.code("".join(traceback.format_exception(*sys.exc_info())))
else:
    st.info("Set your values in the sidebar and click **Run RAG**.")

# (Optional) quick health check on import to catch misconfig early
def _validate_config():
    missing = []
    if "<YOUR-SEARCH-NAME>" in AZURE_SEARCH_ENDPOINT: missing.append("AZURE_SEARCH_ENDPOINT")
    if AZURE_SEARCH_INDEX == "<YOUR-INDEX-NAME>":    missing.append("AZURE_SEARCH_INDEX")
    if AZURE_SEARCH_API_KEY == "<YOUR-SEARCH-ADMIN-KEY>": missing.append("AZURE_SEARCH_API_KEY")
    if "<YOUR-AOAI-NAME>" in AZURE_OPENAI_ENDPOINT:  missing.append("AZURE_OPENAI_ENDPOINT")
    if AZURE_OPENAI_API_KEY == "<YOUR-AOAI-KEY>":    missing.append("AZURE_OPENAI_API_KEY")
    if missing:
        st.warning("Fill these config values before running: " + ", ".join(missing))

_validate_config()
