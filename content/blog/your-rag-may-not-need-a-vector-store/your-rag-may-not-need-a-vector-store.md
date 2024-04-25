---
title: Using LLM tool calling and long context for better RAG
date: 2024-04-25
tags:
---

When building a RAG pipeline you'll probably reach for a vector store to store embeddings of document chunks, which are then retrieved and put into context at query time. This works well if your users are asking single fact queries where the answer can be found in a relevant document chunk. But if your users want to ask more complex questions where the answer requires information spread across the whole document or across multiple documents, retreiveing chunks often leaves out critical information and can lead to inaccurate responses.

Relying on document chunks has been a great solution to add knowledge to LLMs with a limited context window. But context windows have grown massively over the past year, with the leading LLMs supporting context windows reaching 1M tokens. This opens the door to new approaches to RAG which are less constrained by context.

## Whole document querying RAG

Instead of retrieving document chunks, I've had success retreiving and querying whole documents. Queries like 'summarize xyz document ' or 'compare document abc to xyz' yield a full and complete summary without risk of missing important details.

When does this appraoch work? This approach works best if your documents are all of the same type or can be put into categories, and if the user queries include enough information to locate the specific document(s) the question is for.

For example, if your documents are client contracts, each may have a client name, date and contract type. If a user asks 'Summarize the most recent contract with Acme Inc?' we have enough information to find this document, and then use the whole document as context to fully answer their question.

Querying whole documents like this calls for a different RAG workflow than the common single step chunk-retrieve-query workflow. Retrieving whole documents and putting them straight into the context could fill up even a large context window.

Instead, we can leverage the function/tool calling ability of many LLMs to create sub-queries to query each document, which can be executed in parallel. We can even make use of cheaper and faster LLMs for these sub-queries which have to process the complete documents.

What does this look like in practice?

### Create document query functions

In the client contracts example, we would need to be able to locate and query a client contract document. We can create a function which takes several search filters, retrieves the full text of the top matching document, and then calls an LLM (e.g. gpt-3.5-turbo) with the full document text and the query. The fuction should accept the filters required to find the document e.g.: client name, date range, contract type. Plus a query param which is the query to send to the LLM with the full document text.

There's no set way to search for these documents, you could use SQL, Elastic or even embeddings. The key thing is it should be able handle fuzzy search filters for certain params, e.g. for the client name in this case.

Here's an example of this function in Python:

```python
def query_client_contract(client_name: str, document_type: str, from_date: str = None, to_date: str = None, query: str):
	# Search for the document
	document = search_client_contract(client_name, document_type, from_date, to_date)
	# Call the LLM with the full document text and the query
    messages = [
        {"content": "Answer the query using the provided text.", "role": "system"},
        {"content": document + "\n\nQuery: " + query, "role": "user"},
    ]
	response = client.chat.completions.create(
        model="gpt-3.5-turbo", # Use a cheaper model for the sub-query which will process the full document
        messages=messages,
    )
	return response.choices[0].message.content
```

### Sub-query function calls

Now we have the document query function, we are going to use [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) to create sub-queries to this function.

First we use JSON Schema to define the tool for OpenAI function calling:

```python
tools = [
	{
		"type": "function",
		"function": {
			"name": "query_client_contract",
			"description":
				"Send the query to AI to ask the full document text. The AI response will be returned.",
			"parameters": {
				"type": "object",
				"properties": {
					"client_name": {
						"type": "string",
						"description": "Name of the client the contract is for.",
					},
					"document_type": {
						"type": "string",
						"enum": ["contract", "lease"],
						"description": "The type of legal contract.",
					},
					"from_date": {
						"type": "string",
						"format": "date-time",
						"description": "Find documents from this date.",
					},
					"to_date": {
						"type": "string",
						"format": "date-time",
						'description': "Find documents up to this date.",
					},
				},
				"required": ["client_name", "document_type"],
			},
		},
	}
]
```

Then we need create a helper function to execute the function when requested by the LLM:

```python
def execute_function_call(message):
    if message.tool_calls[0].function.name == "query_client_contract":
        args = json.loads(message.tool_calls[0].function.arguments)
        results = ask_database(args["client_name"], args["document_type"], args["from_date"], args["to_date"], args["query"])
    else:
        results = f"Error: function {message.tool_calls[0].function.name} does not exist"
    return results
```

Now in the main chat function, we take a user's query, and if GPT suggests a function call, we execute it and append the results to the chat messages, and then send the messages back to GPT for the final answer:

```python
def ask_ai(query: str):
    messages = [
        {"content": "Answer the user query, calling functions if required.", "role": "system"},
        {"content": query, "role": "user"},
    ]

	chat_response = client.chat.completions.create(
        model="gpt-4-turbo", # Use a more powerful model for function calling
        tools=tools,
        tool_choice="auto", # "auto" means the model can pick between generating a message or calling a function
        messages=messages,
    )

	assistant_message = chat_response.choices[0].message
	assistant_message.content = str(assistant_message.tool_calls[0].function)
	messages.append({"role": assistant_message.role, "content": assistant_message.content})

	if assistant_message.tool_calls:
		results = execute_function_call(assistant_message)
		messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})

	second_chat_response = client.chat.completions.create(
        model="gpt-4-turbo", # Use a more powerful model for function calling
        tools=tools,
        tool_choice="auto", # "auto" means the model can pick between generating a message or calling a function
        messages=messages,
    )
	print(second_chat_response.choices[0].message.content)
```

## The benefits of this approach

There are several benefits to this approach. The main benefit, as discussed above, is that we are querying whole documents. For many use cases this is going to provide more complete answers for users. You can also easily extend this approach by adding more functions for different document types and data sources. GPT will call multiple functions which you can execute in parallel, and in the final GPT call we can use gpt-4-turbo to integrate the results and provide the final answer. If you do have a handful of unknown documents, you can still use the chunk-retrieve-query approach for those, and simply add a function to the tool list to query the chunked documents with a typical RAG pipeline.

I'm excited to see how this approach can be used in practice. I think it will be especially useful for complex questions where the answer is spread across multiple documents, or where the user query is for a summary of a document. I'd love to hear how you get on with this approach. Please reach out if you have any other ideas for how to improve this approach, or related new ideas for improving RAG.
