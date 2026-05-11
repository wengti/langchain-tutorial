import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from openai.types import vector_store
from pinecone import Pinecone

load_dotenv()

if __name__ == "__main__":

    # Load Data
    text_loader = TextLoader(file_path="mediumblog1.txt", encoding="UTF-8")
    document = text_loader.load()

    # Split Data
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    document_chunks = text_splitter.split_documents(document)
    print(len(document_chunks))

    # Embed Data
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Upload / Store Data
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("INDEX_NAME"))
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    uuids = [str(uuid4()) for _ in range(len(document_chunks))]
    vector_store.add_documents(documents=document_chunks, ids=uuids)
