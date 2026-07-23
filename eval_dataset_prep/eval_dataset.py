# from ragas.testset import TestsetGenerator
# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.documents import Document
# from langsmith import Client
from dotenv import load_dotenv
import os
import pandas as pd
from langsmith import Client
load_dotenv()

# api_key = os.getenv("GEMINI_APIKEY")
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")

# client = Client()

# generator_llm = LangchainLLMWrapper(
#     ChatGoogleGenerativeAI(
#         model="gemini-3.6-flash",
#         temperature=0,
#         api_key = api_key
#     )
# )

# embeddings = LangchainEmbeddingsWrapper(
#     GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001", api_key = api_key
#     )
# )


# generator = TestsetGenerator(
#     llm=generator_llm,
#     embedding_model=embeddings,
# )

# vector_store = Chroma(
#     persist_directory="./vector_store/chroma_db",
#     collection_name="fHF22Wxuyw4",
#     embedding_function=embeddings
# )

# docs = vector_store.get()

# documents = [
#     Document(
#         page_content=text,
#         metadata=metadata
#     )
#     for text, metadata in zip(
#         docs["documents"],
#         docs["metadatas"]
#     )
# ]

# testset = generator.generate_with_langchain_docs(
#     documents,
#     testset_size=30
# )

# df = testset.to_pandas()

# df["video_id"] = "fHF22Wxuyw4"
# df.to_csv("youtube_eval_dataset.csv", index=False)
# print("Saved to youtube_eval_dataset.csv")
df = pd.read_csv("C:/Users/vraja/OneDrive/Desktop/Placements/youtube_assistant/youtube_eval_dataset.csv")
client = Client()

dataset = client.upload_csv(
    csv_file="youtube_eval_dataset.csv",
    input_keys=["user_input"],
    output_keys=["reference"],
    name="YouTube QA Eval Dataset",
    description="Ragas-generated eval set for the YouTube transcript QA chatbot (fHF22Wxuyw4)",
    data_type="kv"
)

print(f"Dataset uploaded: {dataset.name}, id={dataset.id}")
