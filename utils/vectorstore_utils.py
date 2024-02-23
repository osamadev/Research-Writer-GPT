import pinecone
import streamlit as st 

# Function to initialize Pinecone
def initialize_pinecone():
    index_name = 'research-assistant'
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    pinecone_env = st.secrets["PINECONE_ENV"]

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric='cosine', dimension=1536)

    index = pinecone.Index(index_name)
    return index
