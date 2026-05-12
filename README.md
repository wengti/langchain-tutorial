# LANGCHAIN TUTORIAL
* Learning Source: https://www.udemy.com/course/langchain/


## Course Coverage
* 1.0-Introduction: 
    * Basic LCEL

* 2.0-Search-Agent:
    * LLM agent with tools
    * Tavily Search
    * Pydantic (Base Model, Field)

* 3.0-Agent-Under-The-Hood:
    * ReAct (Reasoning and Action) Loop

* 4.0-RAG
    * RAG using information from plain text file
    * RAG LCEL chain
    * Load Data -> Split Data -> Embed Data -> Store Data
    * Pinecone as vector database

* 5.0-RAG-Documentation-Helper
    * Deployment: https://rag-nextjs-assistant.streamlit.app/
    * RAG using information scraped from Internet using Tavily
    * A list of hyperparameter and its value for data scraping and storage
    * Using `asyncio` to perform tasks in parallel in scraping and storing data
    * Using `response_format="content_and_artifact"` to store the source of AI's claim source for frontend display
    * RAG + ReAct Loop (RAG as a tool at the agent's disposal and use when it thinks needed)
    * Frontend Chat Interface using Streamlit


## Retrieval-Augmented Generation (RAG) with LangChain Express Language (LCEL)

### General / Important Concept to know about LCEL
1. `|` Pipe Operator
```python
pipe = prompt | model
```
* Using pipe operator, it parses the output from a `Runnable` Interaface to another `Runnable` Interface, forming a full pipeline.
* When the pipeline is called, it can be called as following: `pipe.invoke()` or `pipe.stream()`, essentially making each of the component within the pipeline to conduct `invoke` and then pass the output to call `invoke` in the next component.

### Explaining how to use LCEL to create a RAG chain / pipeline
* Full code can be found in `4.0-rag/main.py`
* Since the code is called with `.invoke()`, it is important to understand what are the parameters required by  `.invoke()` in each of the following components in the pipeline.
```python
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
```

1. prompt_template
```python
prompt_template = PromptTemplate.from_template(
    "Answer the following questions based on the provided context.\n"
    "Context: {context} \n"
    "Question: {question} \n"
)
```
* For prompt_template, when it is invoked, it is expecting the following: `prompt_template.invoke({'context': ..., 'question':...})`

```python
chain.invoke({"question": "What is Pinecone?"})
```
* When the entire chain is invoked, an input of `dict` containign the question is already provided, so that `question` value is fulfilled
* Using `RunnablePassthrough`, it allows us to have a `Runnable` interface to add extra key/value to the dictionary.
* Read more about `RunnablePassthrough`: https://reference.langchain.com/python/langchain-core/runnables/passthrough/RunnablePassthrough

```python
context=RunnableLambda(lambda x: x["question"])
| retriever
| RunnableLambda(lambda x: "\n\n".join([item.page_content for item in x]))
```
* since retriever when it is invoked, it is expecting the following: `retriever.invoke(value: str)`
* `RunnableLambda(lambda x: x["question"])` is used to extract the value to pass into `retriever.invoke()`
* `RunnableLambda(lambda x: "\n\n".join([item.page_content for item in x]))` is done to join the returned result into 1 string and assign as 'context' value for the dictionary.
* Finally the this output dictionary consists of both `context` and `question`, which is then fed into the prompt template

2. llm_agent
* Because llm_agent is expecting the following when invoked: `llm_agent.invoke({"message": [HumanMessage(content="..."), AIMessage(content="...")]})`
* `RunnableLambda(lambda x: {"messages": x.to_messages()})` can effectively convert the output of prompt template into this form.
* `RunnableLambda(lambda x: x["messages"][-1].content)` then finally extract the content of the reply from the agent.


### What is the advantages of using LCEL?
* Built-in streaming
* Type-Safety
* Reusable Chain
* Better Debugging with LangSmith (all operations are stored under one chain)


## Scraping Data using Tavily Crawl (or Tavily Map + Tavily Extract)

### Used Hyperparmeter as a starting point
```python
# Crawling
CRAWL_MAX_DEPTH = 5  # Maximum depth from the source URL to be crawled
CRAWL_MAX_BREADTH = 20  # Maximum number of pages to be crawled, branching out from each site
CRAWL_LIMIT = 200  # Maximum total number of pages to be crawled

# Extracing
EXTRACT_CHUNK_SIZE = 20  # Number of URLs to be extracted for each API hit

# Text Splitter
TEXT_SPLITTER_CHUNK_SIZE = 4000
TEXT_SPLITTER_CHUNK_OVERLAP = 200

# Embedding Model
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_CHUNK_SIZE = 50  
EMBEDDING_RETRY_MIN_SECONDS = 10

# Vector Database
DATABASE_CHUNK_SIZE = 200 # Amount of chunks to be sent to be embedded and uploaded to database at once
```

### Embed and Store Vector Embeddings in parallel
```python
tasks = [
    embed_and_add_one_batch(document, idx)
    for idx, document in enumerate(batched_documents)
]
await asyncio.gather(*tasks, return_exceptions=True)
```
* `embed_and_one_batch()` is an asynchronous function

```python
if __name__ == "__main__":
    asyncio.run(main())
```
* To run a function asynchronously


### Get citation for AI claimed sources
* Make sure that for each split chunk, the source of the chunk is stored in the database
```python
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
```
* `response_format="content_and_artifact"` expecting that the tools return a tuple of (content, artifact)
    * where content will be a string for the LLM
    * artifact is not provided to the LLM for the remainder ReAct Loop but available for frontend display

```python
sources = []
for message in response["messages"]:
    if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
        for artifact in message.artifact:
            sources.append(artifact.metadata.get("source", "unknown"))
```
* After receiving a response from the LLM agent, we can loop through all the messages and target the Tool Message and gather all the artifacts to store them.
