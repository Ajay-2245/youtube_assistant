from youtube_transcript_api import YouTubeTranscriptApi
from utility_functions.functions import async_language_converter, chunker, vector_store_persist, retrieve_relavant_context, load_existing_vector_store
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import asyncio
import chromadb

load_dotenv()
gemini_api_key = os.getenv("GEMINI_APIKEY")
ytt_api = YouTubeTranscriptApi()



@tool
def transcript_context_extractor(video_id:str, query:str)->str:
    """Takes a youtube video url,
    gets the transcript of the video as a string,
    loads the context using document loader, chunks it, stores the chunks in vector store,
    based on the query returns the relavant chunks, aggregates the relavant chunks and gives
    final context string.
    """
    context = ""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(base_dir, "..", "vector_store", "chroma_db")

    #Fetch all the already available embeddings
    client = chromadb.PersistentClient(path="./vector_store/chroma_db")
    collections = client.list_collections()
    #see if vector store already has the embeddings of the video
    flag = False
    for collection in collections:
        if video_id == collection.name:
            flag = True
            break
    if flag:
        vector_store = load_existing_vector_store(persist_dir, video_id)
    else :
        #fetch the transcript
        if video_id:
            try:
                transcript = ytt_api.fetch(video_id, languages=["en", "hi"], preserve_formatting=True)
                for snippet in transcript:
                    context+=snippet.text
            except Exception as e:
                print("Transcript API call failed")
                print(e)
        else :
            print("not able to fetch video id")
            return ""        
        
        #Perform semantic chunking on the raw transcript
        raw_chunks = chunker(context)

        #convert the language to english if it is in hindi
        translated_chunks = asyncio.run(async_language_converter(raw_chunks, video_id)) 


        #create the persist directory
        os.makedirs(persist_dir, exist_ok=True)

        #save in vector store
        vector_store = vector_store_persist(translated_chunks, persist_dir, video_id)

    #retrieve relavant docs
    final_context = retrieve_relavant_context(query, vector_store, 5)
    print(final_context)
    
    return final_context

ans  = transcript_context_extractor.invoke({"video_id" : "fHF22Wxuyw4", "query" : "What is difference between ML and DL?"})
print(ans)