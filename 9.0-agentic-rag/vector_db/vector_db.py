import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# Hyperparameter
EMBEDDING_RETRY_MIN_SECONDS = 10


def create_vector_store():
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_NAME"),
        retry_min_seconds=EMBEDDING_RETRY_MIN_SECONDS,
    )
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("INDEX_NAME"))
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return vector_store
