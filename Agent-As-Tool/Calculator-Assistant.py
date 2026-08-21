"""
Pattern: Agent as Tool
Project: ToolBot - AI Calculator Assistant

An agent that uses Python functions as tools. The LLM decides which
tool to call, the tool executes, and the result is formatted back.

This demonstrates how agents can use tools to perform actions they
can't do alone (like precise math calculations).

Uses LangGraph's chained builder with sequential tool-use flow.
"""

import math
import json
from typing import TypedDict

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph


# -------------------------
# State
# -------------------------

class State(TypedDict):
    question: str
    tool_call: str
    tool_result: str
    answer: str


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Available Tools (Python functions)
# -------------------------

def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> str | float:
    if b == 0:
        return "Error: Division by zero"
    return a / b

def power(base: float, exponent: float) -> float:
    return base ** exponent

def sqrt(a: float) -> float:
    return math.sqrt(a)


TOOLS = {"add": add, "subtract": subtract, "multiply": multiply,
         "divide": divide, "power": power, "sqrt": sqrt}

TOOL_DESCRIPTIONS = """Available tools:
- add(a, b): Add two numbers
- subtract(a, b): Subtract b from a
- multiply(a, b): Multiply two numbers
- divide(a, b): Divide a by b
- power(base, exponent): Raise base to exponent
- sqrt(a): Square root of a number"""


# -------------------------
# Nodes
# -------------------------

def decide_tool(state: State):
    """LLM decides which tool to call and with what arguments."""
    response = llm.invoke(
        f"You are a calculator assistant. Decide which tool to use.\n\n"
        f"{TOOL_DESCRIPTIONS}\n\n"
        f"User question: {state['question']}\n\n"
        f'Reply with ONLY a JSON object: {{"tool": "add", "args": [5, 3]}}\n'
        f"If no tool is needed, reply: {{}}"
    )
    return {"tool_call": response.content.strip()}


def execute_tool(state: State):
    """Executes the chosen tool with the given arguments."""
    try:
        call = json.loads(state["tool_call"])
        if not call or "tool" not in call:
            return {"tool_result": "No calculation needed."}

        tool_name = call["tool"]
        args = call.get("args", [])

        if tool_name not in TOOLS:
            return {"tool_result": f"Unknown tool: {tool_name}"}

        result = TOOLS[tool_name](*args)
        return {"tool_result": str(result)}
    except (json.JSONDecodeError, TypeError) as e:
        return {"tool_result": f"Error: {e}"}


def format_answer(state: State):
    """Formats the tool result into a natural language answer."""
    response = llm.invoke(
        f"The user asked: {state['question']}\n"
        f"The calculation result is: {state['tool_result']}\n\n"
        f"Provide a short, clear answer."
    )
    return {"answer": response.content}


# -------------------------
# Build Graph (chained builder)
# -------------------------

graph = (
    StateGraph(State)
    .add_node(decide_tool)
    .add_node(execute_tool)
    .add_node(format_answer)
    .add_edge(START, "decide_tool")
    .add_edge("decide_tool", "execute_tool")
    .add_edge("execute_tool", "format_answer")
    .add_edge("format_answer", END)
    .compile()
)


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    print("ToolBot - AI Calculator Assistant")
    print("Ask a math question (or 'exit' to quit):\n")

    while True:
        question = input("> ")
        if question.lower() == "exit":
            break

        result = graph.invoke({"question": question})
        print(f"\n{result['answer']}\n")
