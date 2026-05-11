import os

from dotenv import load_dotenv

from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Load environment variables
load_dotenv()

# Setup database
embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL_NAME"),
    chunk_size=os.getenv("EMBEDDING_CHUNK_SIZE"),
    retry_max_seconds=os.getenv("EMBEDDING_RETRY_MIN_SECONDS"),
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# Creating tools
@tool(response_format="content_and_artifact")
def retrieval_tool(query: str):
    """A tool that is used to get relevant context from a vector database to ground the answer to the user.
    Args:
        query: a string that is used to get the relevant context to help answering the questions.
    Returns:
        A string that contains all the retrieved context.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    results = retriever.invoke(query)
    formatted_results = "\n\n".join(
        [
            f"Source:{result.metadata.get("source", "unknown")}\nContent: {result.page_content}"
            for result in results
        ]
    )
    return formatted_results, results


# Creating an agent
tools = [retrieval_tool]
system_prompt = (
    "You are a helpful AI assistant that will answer users' queries amicably.\n"
    "You have access to a vector database containing documentations of NextJS.\n"
    "When a user queries anything about NextJS, try to get relevant context using the provided tools. \n"
    "If none of the context is helpful for answering the question, please say so and ask the user for further assistance or clarification. \n"
    "Do not try to answer a NextJS question without basing your answer on the retrieved context from the tool."
)

llm_agent = create_agent(
    model="openai:gpt-5.4-mini",
    tools=tools,
    system_prompt=system_prompt,
)


def chat_with_llm(question: str):
    response = llm_agent.invoke({"messages": [HumanMessage(question)]})
    answer = response["messages"][-1].content

    sources = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            for artifact in message.artifact:
                sources.append(artifact.metadata.get("source", "unknown"))

    # Remove duplication
    sources = list(set(sources))

    return answer, sources


if __name__ == "__main__":
    answer, sources = chat_with_llm(
        question="What is the best practice in fetching the data in NextJS?"
    )
    print(f"AI's answer: {answer}")
    print(f"Sources: ")
    for s in sources:
        print(s)
