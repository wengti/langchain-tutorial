import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL_NAME"),
    chunk_size=os.getenv("EMBEDDING_CHUNK_SIZE"),
    retry_max_seconds=os.getenv("EMBEDDING_RETRY_MIN_SECONDS"),
)
single_vector = embeddings.embed_query("Google Font")
print(single_vector)
