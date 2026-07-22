import chromadb
import os
from utility_functions.functions import async_language_converter, chunker, vector_store_persist, retrieve_relavant_context, load_existing_vector_store

base_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(base_dir, "vector_store", "chroma_db")

client = chromadb.PersistentClient(path="./vector_store/chroma_db")
collections = client.list_collections()
#see if vector store already has the embeddings of the video
flag = False
for collection in collections:
    if "fHF22Wxuyw4" == collection.name:
        flag = True
        break
if flag:
    vector_store = load_existing_vector_store(persist_dir, "fHF22Wxuyw4")