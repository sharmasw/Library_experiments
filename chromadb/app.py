import streamlit as st

st.title("My quick Search App")

title = st.text_input("Find me the email", "Life of Brian")
st.write("What you searched", title)

import chromadb
from chromadb.config import  Settings
chroma_client = chromadb.PersistentClient(
    path = 'db/',
    settings = Settings()
)

from chromadb.utils import embedding_functions
model_name = "all-MiniLM-L6-v2"
emb_f = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

collection2 = chroma_client.get_or_create_collection(name="newspaper_email_col",embedding_function=emb_f)

results = collection2.query(
    query_texts=[title], # Chroma will embed this for you
    n_results=10 # how many results to return
)
print(results)

import pandas as pd

resa=pd.DataFrame(results['documents'])

st.write(resa.transpose())
