import os
import streamlit as st
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv()

# ---------- CONFIG (fill via .env or inline) ----------
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://<your-search>.search.windows.net")
SEARCH_INDEX    = os.getenv("AZURE_SEARCH_INDEX", "<your-index>")
SEARCH_API_KEY  = os.getenv("AZURE_SEARCH_API_KEY", "<your-search-key>")
SEMANTIC_CONFIG = os.getenv("SEMANTIC_CONFIG_NAME", "default")

# Fields from the “Import and Vectorize” wizard
CHUNK_FIELD   = os.getenv("CHUNK_TEXT_FIELD", "chunk")
VECTOR_FIELD  = os.getenv("CHUNK_VECTOR_FIELD", "text_vector")

AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://<your-aoai>.openai.azure.com/")
AOAI_KEY      = os.getenv("AZURE_OPENAI_API_KEY", "<your-aoai-key>")
EMB_DEPLOY    = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
CHAT_DEPLOY   = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
AOAI_API_VER  = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# ---------- CLIENTS ----------
search_client = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, AzureKeyCredential(SEARCH_API_KEY))
aoai = AzureOpenAI(azure_endpoint=AOAI_ENDPOINT, api_key=AOAI_KEY, api_version=AOAI_API_VER)

# ---------- STEP 1: embed + HYBRID search (semantic + vector) ----------
def get_embedding(text: str) -> list[float]:
    emb = aoai.embeddings.create(model=EMB_DEPLOY, input=text)
    return emb.data[0].embedding

def hybrid_search(user_query: str, k: int = 10):
    qvec = get_embedding(user_query)  # ← Step 1 requires this
    results = search_client.search(
        search_text=user_query,                 # semantic boost (optional but recommended)
        query_type="semantic",
        semantic_configuration_name=SEMANTIC_CONFIG,
        vector_queries=[{
            "vector": qvec,
            "k": k,
            "fields": VECTOR_FIELD,
            "kind": "vector"
        }],
        select="*",
        include_total_count=True,
        query_caption="extractive",
        query_caption_highlight_enabled=True,
    )
    docs = list(results)
    context = "\n\n".join([d.get(CHUNK_FIELD, "") for d in docs])  # build RAG context from 'chunk'
    return docs, context, results
# (Matches your PDF’s Step 1: use embeddings + vector_queries + chunk field.)  # :contentReference[oaicite:3]{index=3}

# ---------- STEP 2: generate the answer from the retrieved context ----------
def answer_from_context(context: str, question: str) -> str:
    prompt = f"""Use the context below to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
    resp = aoai.chat.completions.create(
        model=CHAT_DEPLOY,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip()
# (This is your PDF’s Step 2.)  # :contentReference[oaicite:4]{index=4}

# ---------- STREAMLIT UI (kept like your sample) ----------
query_text = st.text_input("Your query", value="What is the capital of France?")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Filtered Response", "First @search.answers", "First Chunk", "OpenAI Response"]
)

if query_text:
    docs, context, results = hybrid_search(query_text)

    with tab1:
        st.write("Search docs (top few):")
        st.json(docs[:3])

    with tab2:
        # In most SDK versions, 'answers' are on the overall results, not each doc.
        # If your SDK exposes them, show them; otherwise fall back gracefully.
        try:
            answers = results.get_answers()  # may not exist depending on package version
            if answers:
                st.write("First @search.answers:", answers[0].text)
            else:
                st.write("No @search.answers found.")
        except Exception:
            st.info("Answers not available in this SDK version; skipping.")

    with tab3:
        if docs and CHUNK_FIELD in docs[0]:
            st.write("First Chunk:", docs[0][CHUNK_FIELD])
        else:
            st.write(f"No '{CHUNK_FIELD}' found—check your index field names.")

    with tab4:
        if context.strip():
            st.write("OpenAI Response:", answer_from_context(context, query_text))
        else:
            st.write("No chunk context retrieved from Search.")
