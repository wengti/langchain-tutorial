import os

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pinecone import Pinecone

load_dotenv()


def create_lcel_rag_chain():

    # Create a context retriever
    print("Creating Context Retriever")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("INDEX_NAME"))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    # Create a prompt template
    print("Creating a prompt template")
    prompt_template = PromptTemplate.from_template(
        "Answer the following questions based on the provided context.\n"
        "Context: {context} \n"
        "Question: {question} \n"
    )

    # Create agent
    print("Creating an agent")
    llm_agent = create_agent(
        model="openai:gpt-5.4-mini",
        system_prompt="You are a helpful assistant that answers based on facts and context.",
    )

    # Create a rag chain
    print("Creating a RAG chain")
    rag_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: x["question"])
            | retriever
            | RunnableLambda(lambda x: "\n\n".join([item.page_content for item in x]))
        )
        | prompt_template
        | RunnableLambda(lambda x: {"messages": x.to_messages()})
        | llm_agent
        | RunnableLambda(lambda x: x["messages"][-1].content)
    )
    return rag_chain


def main():
    print("Hello from 4-0-rag!")
    chain = create_lcel_rag_chain()
    answer = chain.invoke({"question": "What is Pinecone?"})
    print(answer)


if __name__ == "__main__":
    main()
