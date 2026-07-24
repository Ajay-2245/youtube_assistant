from youtube_transcript_api import YouTubeTranscriptApi
from utility_functions.functions import async_language_converter, pre_translation_chunker, post_translation_chunker,  vector_store_persist, retrieve_relavant_context, load_existing_vector_store, translation_llm, translation_prompt_version
from langchain_core.tools import tool
from dotenv import load_dotenv
from pathlib import Path
import os
import asyncio
import chromadb
import hashlib
import sqlite3
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
    translated_context = None
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(base_dir, "..", "vector_store", "chroma_db")

    # see if vector store already has the embeddings of the video
    client = chromadb.PersistentClient(path="./vector_store/chroma_db")
    collections = client.list_collections()
    flag = False
    for collection in collections:
        if video_id == collection.name:
            flag = True
            break
        
    if flag:
        ## load existing vector store 
        vector_store = load_existing_vector_store(persist_dir, video_id)

    else :

        # connect to sqlite server
        db_path = Path(__file__).parent.parent / "sqlite" / "transcripts.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        #prepare string that can be hashed 
        temp_string = f"{video_id}|{translation_llm}|{translation_prompt_version}"
        transcript_hash = hashlib.sha256(temp_string.encode("utf-8")).hexdigest()

        #check if the hash already exists in sqlite
        cursor.execute("""
        SELECT transcript_context
        FROM transcript_contexts
        WHERE transcript_hash = ?
        """, (transcript_hash,))

        row = cursor.fetchone()

        if row is not None:
            translated_context = row[0]
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
            raw_chunks = pre_translation_chunker(context)
            print("Pre Translation chunking done....")

            #convert the language to english if it is in hindi
            translated_context = asyncio.run(async_language_converter(raw_chunks))
            print("translation done....")
            print("caching to sqlite....")
            #cache it in sqlite 

            cursor.execute("""
            INSERT INTO transcript_contexts (
                video_id,
                translation_model,
                prompt_version,
                transcript_hash,
                transcript_context
            )
            VALUES (?, ?, ?, ?, ?)
            """, (
                video_id,
                translation_llm,
                translation_prompt_version,
                transcript_hash,
                translated_context
            ))

            conn.commit()
            print("sqlite caching done....Ready for retrieval")

        #have to evaluate post_translation_chunker not this entire function(the chunker contains all the chunking variables so can be evaluated thoroughly)
        #chunk the translated_context
        #Fetch all the already available embeddings


        ##perform chunking and persist in vector store
        translated_chunks = post_translation_chunker(translated_context, video_id)
        print("")
        #create the persist directory
        os.makedirs(persist_dir, exist_ok=True)

        #save in vector store
        vector_store = vector_store_persist(translated_chunks, persist_dir, video_id)

    #retrieve relavant docs
    final_context = retrieve_relavant_context(query, vector_store, 5)
    print(final_context)
    
    return final_context

# ans  = transcript_context_extractor.invoke({"video_id" : "QQfZAoNGQmE", "query" : "What is the motivation of using GRU?"})
# print(ans)