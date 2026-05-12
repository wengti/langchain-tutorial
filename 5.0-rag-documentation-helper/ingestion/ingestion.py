import os
from typing import List
from uuid import uuid4

from dotenv import load_dotenv
import asyncio
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# ==================================================== #
# Hyperparameter
# Crawling
CRAWL_MAX_DEPTH = 5  # Maximum depth from the source URL to be crawled
CRAWL_MAX_BREADTH = (
    20  # Maximum number of pages to be crawled, branching out from each site
)
CRAWL_LIMIT = 200  # Maximum total number of pages to be crawled

# Extracing
EXTRACT_CHUNK_SIZE = 20  # Number of URLs to be extracted for each API hit

# Text Splitter
TEXT_SPLITTER_CHUNK_SIZE = 4000
TEXT_SPLITTER_CHUNK_OVERLAP = 200


# Embedding Model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_CHUNK_SIZE = os.getenv("EMBEDDING_CHUNK_SIZE")
EMBEDDING_RETRY_MIN_SECONDS = os.getenv("EMBEDDING_RETRY_MIN_SECONDS")

# Vector Database
DATABASE_CHUNK_SIZE = (
    200  # Amount of chunks to be sent to be embedded and uploaded to database at once
)
# ==================================================== #

# Create needed tools
tavily_map = TavilyMap()

# Create database and its embedding tool
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    chunk_size=EMBEDDING_CHUNK_SIZE,
    retry_max_seconds=EMBEDDING_RETRY_MIN_SECONDS,
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME"))


# Crawl Data - Can also be done using Tavily Crawl - here is just demo for using map + extract
# Extract Data
async def extract_data(url_list: List[str], index: int):
    try:
        print(f"[INFO] Extracting batch {index}\n")
        tavily_extract_tool = TavilyExtract(extract_depth="advanced")
        result = await tavily_extract_tool.ainvoke({"urls": url_list})
        print(f"[Success] Batch {index} is successfully extracted.\n")
        return result
    except Exception as err:
        print(f"[Failure] Batch {index} fails to be extracted!\n")
        return err


# Crawl Data
async def crawl_data():
    # Get urls branching out from the root
    mapped_results = await tavily_map.ainvoke(
        {
            "url": "https://nextjs.org/docs",
            "max_depth": CRAWL_MAX_DEPTH,  # Start low to test things out
            "max_breadth": CRAWL_MAX_BREADTH,
            "limit": CRAWL_LIMIT,
        }
    )
    urls = mapped_results["results"]
    print(f"A total of {len(urls)} site are to be crawled.\n")

    # Breaking up the full list of url into a list of list with 5 urls each
    chunked_urls = [
        urls[i : i + EXTRACT_CHUNK_SIZE]
        for i in range(0, len(urls), EXTRACT_CHUNK_SIZE)
    ]

    # Extract concurrently for each of the url chunks
    tasks = [
        extract_data(chunked_url, chunked_idx)
        for chunked_idx, chunked_url in enumerate(chunked_urls)
    ]
    extracted_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine them back into one huge list
    combined_results = []
    for item in extracted_results:
        combined_results.extend(item["results"])  # {url: , title: , raw_content: }

    return combined_results


# Embedding and Add to Database
async def embed_and_add_to_db(documents: List[Document], batch_size: int):
    batched_documents = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    async def embed_and_add_one_batch(batched_document: List[Document], batch_idx: int):
        try:
            print(f"[INFO] Embedding and Updating for batch {batch_idx}\n")
            vector_store = PineconeVectorStore(index=index, embedding=embeddings)
            uuids = [str(uuid4()) for _ in range(len(batched_document))]
            await vector_store.aadd_documents(documents=batched_document, ids=uuids)
            print(f"[SUCCESS] Batch {batch_idx} has been successfully uploaded. \n")
        except Exception as err:
            print(
                f"[FAILURE] Batch {batch_idx} has failed to be embedded and uploaded. "
            )
            print(f"Error message: {str(err)}")

    tasks = [
        embed_and_add_one_batch(document, idx)
        for idx, document in enumerate(batched_documents)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    # Crawl Data
    print("Crawling data")
    crawled_data = await crawl_data()

    # Split Data
    print("Splitting data")
    all_documents = [
        Document(page_content=data["raw_content"], metadata={"source": data["url"]})
        for data in crawled_data
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CHUNK_SIZE, chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP
    )
    chunked_documents = text_splitter.split_documents(
        all_documents
    )  # {metadata: {source: }, page_content: }
    print(
        f"A total of {len(chunked_documents)} chunks are to be embedded and uploaded. \n"
    )

    # Embed and Store Data
    print("Embedding and storing data")
    await embed_and_add_to_db(
        documents=chunked_documents, batch_size=DATABASE_CHUNK_SIZE
    )


if __name__ == "__main__":
    asyncio.run(main())
