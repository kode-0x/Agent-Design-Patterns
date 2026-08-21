"""
Pattern: Router / Classifier
Project: MailRouter - Email Classifier

A router node classifies incoming emails into categories,
then uses Command to route AND update state in a single step.

Flow: [Classify + Route via Command] -> [Spam | Support | Sales | Personal] handler

Uses LangGraph's Command(update, goto) for combined state update + routing.
"""

from typing import TypedDict, Literal

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command


# -------------------------
# State
# -------------------------

class State(TypedDict):
    email: str
    category: str
    response: str


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Router: Classify and route with Command
# -------------------------

def classify_email(state: State) -> Command[Literal["spam", "support", "sales", "personal"]]:
    """Classifies email and routes to the correct handler in one step."""
    response = llm.invoke(
        f"Classify this email into exactly ONE category: spam, support, sales, or personal.\n\n"
        f"Email: {state['email']}\n\n"
        f"Reply with ONLY the category name (lowercase, one word)."
    )
    category = response.content.strip().lower()

    if category not in ["spam", "support", "sales", "personal"]:
        category = "support"

    # Command updates state AND routes to the next node
    return Command(update={"category": category}, goto=category)


# -------------------------
# Handler Nodes
# -------------------------

def spam(state: State):
    return {"response": "🚫 This email has been flagged as spam and moved to the spam folder."}


def support(state: State):
    response = llm.invoke(
        f"You are a support agent. Draft a helpful reply to this email:\n\n{state['email']}"
    )
    return {"response": f"📋 SUPPORT RESPONSE:\n{response.content}"}


def sales(state: State):
    response = llm.invoke(
        f"You are a sales agent. Draft a professional reply to this inquiry:\n\n{state['email']}"
    )
    return {"response": f"💼 SALES RESPONSE:\n{response.content}"}


def personal(state: State):
    response = llm.invoke(
        f"You are a friendly assistant. Draft a warm reply to this email:\n\n{state['email']}"
    )
    return {"response": f"👤 PERSONAL RESPONSE:\n{response.content}"}


# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(State)

builder.add_node("classify", classify_email)
builder.add_node("spam", spam)
builder.add_node("support", support)
builder.add_node("sales", sales)
builder.add_node("personal", personal)

builder.add_edge(START, "classify")
builder.add_edge("spam", END)
builder.add_edge("support", END)
builder.add_edge("sales", END)
builder.add_edge("personal", END)

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    print("MailRouter - Email Classifier")
    print("Paste your email below (press Enter twice to submit):\n")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    email = "\n".join(lines)

    print("\n⏳ Classifying email...\n")

    result = graph.invoke({"email": email})

    print(f"Category: {result['category'].upper()}")
    print("-" * 40)
    print(result["response"])
