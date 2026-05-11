# LANGCHAIN TUTORIAL
* Learning Source: https://www.udemy.com/course/langchain/

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
