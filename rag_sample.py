import streamlit as st
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import json
import os
import base64
from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv()


# Replace with your actual values with vector search enabled
# Ensure you have the correct endpoint, index name, and API key for your Azure Search service
endpoint = "your-search-endpoint"  # e.g., "https://your-search-service.search.windows.net"
index_name = "your-index-name"
api_key = "your-api-key"
query_text = "your-query-text"  # e.g., "What is the capital of France?"

# Create the SearchClient
search_client = SearchClient(
    endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key)
)

# Perform the search with semantic configuration
results = search_client.search(
    search_text=query_text,
    query_type="semantic",
    query_answer="extractive|count-3",
    query_caption="extractive",
    semantic_configuration_name="your-semantic-configuration-name",  # e.g., "default"
    select="*",
    include_total_count=True,
    query_caption_highlight_enabled=True,
)
# Convert results to a list for easier processing
results = list(results)
for result in results:
    print("test1----", result["chunk"])

# Azure OpenAI Configuration
OPENAI_API_KEY = "your-openai-api-key"
endpoint = os.getenv("ENDPOINT_URL", "your-openai-endpoint")  # e.g., "https://your-openai-service.openai.azure.com/"
deployment = os.getenv("DEPLOYMENT_NAME", "your-deployment-name")  # e.g., "gpt-35-turbo"
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", OPENAI_API_KEY)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

# Display results in Streamlit
tab1, tab2, tab3, tab4 = st.tabs(
    ["Filtered Response", "First @search.answers", "First Chunk", "OpenAI Response"]
)

with tab1:
    # Display the whole response in JSON format
    st.write("Full Response:")
    st.json(results)

with tab2:
    # Display the first "@search.answers" response
    for result in results:
        if "@search.answers" in result and result["@search.answers"]:
            st.write("First @search.answers:", result["@search.answers"][0]["text"])
            break
    else:
        st.write("No @search.answers found in the results.")

with tab3:
    # Display the first "chunk" value
    first_chunk = None
    for result in results:
        if "chunk" in result:
            first_chunk = result["chunk"]
            st.write("First Chunk:", first_chunk)
            break
    else:
        st.write("No chunk found in the results.")

with tab4:
    if first_chunk:
        # Pass the value of result["chunk"] and query_text to Azure OpenAI
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that helps people find information.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{query_text}: Give only {query_text} related Details from {first_chunk}",
                    }
                ],
            },
        ]

        completion = client.chat.completions.create(
            model=deployment,
            messages=chat_prompt,
            max_tokens=1000,
            temperature=0.2,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )

        # Display the deterministic response in Tab 4
        # st.write("OpenAI Response:", completion.to_json())
        st.write("OpenAI Response:", completion.choices[0].message.content)
    else:
        st.write("No chunk available to send to Azure OpenAI.")
 
