import asyncio
import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import api_key, embeddings
from pinecone import Pinecone

from vector_db.vector_db import create_vector_store

# Load environmental variables
load_dotenv()

# Hyperparameter
TEXT_SPLIT_CHUNK_SIZE = 1000
TEXT_SPLIT_OVERLAP_SIZE = 100
SPLIT_DOC_CHUNK_SIZE = 50

# Load
urls = [
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

web_loader = WebBaseLoader(urls)
docs = (
    web_loader.load()
)  # A list of 3 Documents, one for each url Document(metadata, page_content)

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_SPLIT_CHUNK_SIZE, chunk_overlap=TEXT_SPLIT_OVERLAP_SIZE
)
split_docs = text_splitter.split_documents(docs)  # 181 Documents

# Embed and Store
split_docs_chunks = [
    split_docs[i : i + SPLIT_DOC_CHUNK_SIZE]
    for i in range(0, len(split_docs), SPLIT_DOC_CHUNK_SIZE)
]


async def embed_and_add_one_batch(batch_idx, document_chunk):
    try:
        vector_store_instance = create_vector_store()
        print(f"Embedding and Updating batch {batch_idx}")
        uuids = [str(uuid4()) for _ in range(len(document_chunk))]
        await vector_store_instance.aadd_documents(documents=document_chunk, ids=uuids)
        print(f"Completing batch {batch_idx}")
    except Exception as error:
        print(f"Problems with batch {batch_idx}: {str(error)}")


async def main():
    tasks = [
        embed_and_add_one_batch(batch_idx, docs_chunks)
        for batch_idx, docs_chunks in enumerate(split_docs_chunks)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
