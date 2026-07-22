from langchain.agents import create_agent
from tools.transcript import transcript_context_extractor
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv()
gemini_api_key = os.getenv("GEMINI_APIKEY")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key = gemini_api_key)

def build_transcript_agent():

    agent = create_agent(
        model = llm,
        tools=[transcript_context_extractor]
    )
    return agent