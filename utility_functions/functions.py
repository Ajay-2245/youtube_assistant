from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma 
from langsmith import traceable, get_current_run_tree
from dotenv import load_dotenv
import os


load_dotenv()
gemini_api_key = os.getenv("GEMINI_APIKEY")
groq_api_key = os.getenv("GROQ_APIKEY")
ytt_api = YouTubeTranscriptApi()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=gemini_api_key)
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key = gemini_api_key)
translate_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
parser = StrOutputParser()


@traceable(name="llm-language-converter", metadata={"model" : "gemini-3.6-flash", "provider":"Google"})
def language_converter(context):
    prompt = PromptTemplate(
        template = "Convert the following hindi text to english, if it is already in english then give it as it is \n text = {text}",
        input_variables = ['text']
    )
    chain = prompt | llm | parser
    english_context = chain.invoke({'text' : context})

    #create document object for the context
    doc = Document(page_content = english_context)
    docs = [doc]
    return docs

@traceable(name="asynchronous-language-converter", metadata={"model" : "gemini-3.6-flash", "provider":"Google"})
async def async_language_converter(chunks, video_id):
    inputs = [{"text" : chunk} for chunk in chunks]
    prompt = PromptTemplate(
        template = "Convert the following hindi text to english, if it is already in english then give it as it is \n text = {text}",
        input_variables = ['text']
    )
    chain = prompt | llm | parser
    results = await chain.abatch(inputs, config={"max_concurrency":100})

    translated_docs = [
        Document(
            page_content = result,
            metadata = {"video_id" : video_id, "chunk_index":i}
        )
        for i, result in enumerate(results)
    ]
    return translated_docs


@traceable(name="chunker", metadata={"chunking-strategy" : "RecursiveCharacterTextSplitter"})
def chunker(transcript_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap=80)
    chunks = splitter.split_text(transcript_text)
    print(f"{len(chunks)} chunks loaded\n")

    return chunks

@traceable(name="vector-store-persist", metadata={"vector_store" : "ChromaDB", "embedding_model": "models/gemini-embedding-001"})
def vector_store_persist(chunks, persist_dir, video_id):
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = persist_dir,
        collection_name=video_id
    )

    print("Created new embeddings")

    return vector_store

@traceable(name="retrieve-relavant-context", metadata={"strategy" : "similarity-search"})
def retrieve_relavant_context(query, vector_store, num_docs):
    run_tree =  get_current_run_tree()
    if run_tree:
        run_tree.metadata["k"] = num_docs
    results = vector_store.similarity_search(query, k=num_docs)
    final_context = ""
    for doc in results:
        content = doc.page_content
        final_context+=content
    return final_context

@traceable(name="load-vector-store", metadata={"embedding_model" : "models/gemini-embedding-001"})
def load_existing_vector_store(persist_dir, video_id):
    vector_store = Chroma(
            persist_directory = persist_dir,
            embedding_function = embeddings,
            collection_name=video_id
        )
    print("Loaded existing embeddings")
    return vector_store