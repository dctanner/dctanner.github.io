---
title: LLM tool calling as code blocks
date: 2025-02-11
tags:
---

When building sophisticated agents that have more than a handful of tools to call, I've often found the inbuilt structured output/json tool calling methods provided by LMA APIs come up short.

Intuitively, one of the reasons for this is that when the structured output is enabled, there is no room for chain of thought, text or inbuilt support for comments amongst the JSON output of tool calls.

In addition, when you're trying to compare different LLM APIs, you have to switch between different tool calling schemas.

LLMs are trained on a lot of code, and a tool call is really just a function call. So why not just use code blocks for tool calls?

When you use LLMs to output code with tool calls, you may initially think of running the code in a sandbox. But that comes with infra overhead and security concerns. Instead, what if we just parse tool calls in code blocks with regex, and then validate the function names and params before calling the internal functions?

I've had great results with this approach. It's easy to implement and it works with any LLM (including open source models). The chain of thought and comments next to the tool calls is especially helpful when debugging, as the LLM will explain why it decided to call a particular tool with those params.

You can even do clever stuff like parse the text stream from the LLM as it comes in,and call tools as they are returned, instead of waiting for the LLM to finish.

He's an example of this approach that makes use of zod and Vercel AI SDK:

```typescript
// Run this example:
// npm i zod zod-to-ts relaxed-json ai @ai-sdk/openai
// tsx tool-calls-as-ts-example.ts

import { z } from "zod";
import RJSON from "relaxed-json";
import { printNode, zodToTs } from "zod-to-ts";

type ToolsList = {
	[key: string]: { name: string; schema: z.ZodType<unknown> };
};

const getToolsAsTypeScriptString = (toolsList: ToolsList) =>
	Object.entries(toolsList)
		.map(([toolName, { name, schema }]) => {
			const { node } = zodToTs(schema, toolName);
			const nodeString = printNode(node);
			const tsDefAsString = `/** ${name} */ \n${toolName}(${nodeString})`;
			return tsDefAsString;
		})
		.join("\n\n");

const parseToolsCalledContent = ({
	llmResponseWithToolCallsAsJsCodeblock,
	toolsList,
}: {
	llmResponseWithToolCallsAsJsCodeblock: string;
	toolsList: ToolsList;
}) => {
	const toolsCallRegex =
		/(\w+)\(([^()]*(?:\([^()]*\)[^()]*)*)\)(?:\s*\/\/.*)?/g;
	const toolsCalls =
		llmResponseWithToolCallsAsJsCodeblock.matchAll(toolsCallRegex);
	const validatedToolsToCall: {
		name: string;
		args: any;
		originalArgs: string;
	}[] = [];
	for (const match of toolsCalls) {
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		const [_call, toolName, argString] = match;
		// console.log(`Found match for tools call: ${toolsName}(${argString})`)
		if (toolName && toolsList.hasOwnProperty(toolName)) {
			const tool = toolsList[toolName as keyof typeof toolsList];
			const argsObj = RJSON.parse(argString);
			// Validate the arguments using the Zod schema
			const validatedArgs = tool.schema.parse(argsObj);
			validatedToolsToCall.push({
				name: toolName,
				args: validatedArgs,
				originalArgs: argString,
			});
		} else {
			console.warn(`Tool ${toolName} is not found.`);
		}
	}
	return validatedToolsToCall;
};

// EXAMPLE
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
const example = async () => {
	const tools = {
		getWeather: {
			name: "Get weather for location today (default) or N days in the future up to 10 days",
			function: ({
				location,
				daysInFuture,
			}: {
				location: string;
				daysInFuture: number;
			}) => {
				// TODO: Do actualy weather API call
				return {
					location,
					daysInFuture,
					weather: "sunny",
				};
			},
			schema: z.object({
				location: z.string().describe("The location to get the weather for."),
				daysInFuture: z
					.number()
					.describe("The number of days in the future to get the weather for."),
			}),
		},
	};
	const toolsAsTypeScriptString = getToolsAsTypeScriptString(tools);
	const { text: llmResponseWithToolCallsAsJsCodeblock } = await generateText({
		model: openai("gpt-4o"),
		prompt: `
	AVAILABLE_TOOLS:
	"""
    ${toolsAsTypeScriptString}
    """

    AVAILABLE_TOOLS must be called in a single javascript codeblock. All function arguments must be on a single line.

    QUESTION:
    "What is the weather in San Francisco?"
    `,
	});
	console.log("Tools schema pass to llm:\n");
	console.log(toolsAsTypeScriptString);
	console.log("\nResponse from llm with tool call code block:\n");
	console.log(llmResponseWithToolCallsAsJsCodeblock);
	const validatedToolsToCall = parseToolsCalledContent({
		llmResponseWithToolCallsAsJsCodeblock,
		toolsList: tools,
	});
	console.log("\nValidated tools to call:\n");
	console.log(validatedToolsToCall);
};

example();
```

Example output:

````js
$ tsx tool-calls-as-ts-example.ts
Tools schema pass to llm:

/** Get weather for location today (default) or N days in the future up to 10 days */
getWeather({
    /** The location to get the weather for. */
    location: string;
    /** The number of days in the future to get the weather for. */
    daysInFuture: number;
})

Response from llm with tool call code block:

```javascript
getWeather({ location: "San Francisco", daysInFuture: 0 })
```

Validated tools to call:

[
  {
    name: 'getWeather',
    args: { location: 'San Francisco', daysInFuture: 0 },
    originalArgs: '{ location: "San Francisco", daysInFuture: 0 }'
  }
]
````
