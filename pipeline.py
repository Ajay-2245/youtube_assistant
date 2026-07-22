from agents.main import build_transcript_agent
import re


    
def run_youtube_assistant(url:str, query:str)->str:

    pattern = "(?:v=|youtu\.be\/|embed\/|shorts\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    video_id = match.group(1) if match else None
    
    #langsmith config
    langsmith_config = {
        "run_name" : f"transcript_agent_{video_id}",
        "metadata" : {"video_id" : video_id, "query" : query}
    }
    #transcript agent calling
    transcript_agent = build_transcript_agent()
    transcript_agent_result = transcript_agent.invoke({
        "messages" : [("user", f"Extract the query relavant content from the youtube video of id {video_id} and answer the query : {query}, Note : If the query is out of context then  say query is out of available context")]
    }, config = langsmith_config)
    return transcript_agent_result

result = run_youtube_assistant(url="https://www.youtube.com/watch?v=fHF22Wxuyw4&t=75s", query="What is the difference between ML and DL?")
