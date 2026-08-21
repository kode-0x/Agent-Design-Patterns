"""
Pattern: Debate / Multi-Agent
Project: DebateBot - Pros & Cons Generator

Two agents debate a topic in parallel - one argues FOR, the other AGAINST.
A moderator then synthesizes both sides into a balanced conclusion.

Flow: START -> [Pro Agent | Con Agent] (parallel) -> [Moderator] -> END

Uses LangGraph's direct parallel edges from START for fan-out.
"""

from typing import TypedDict

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph


# -------------------------
# State
# -------------------------

class State(TypedDict):
    topic: str
    pro_arguments: str
    con_arguments: str
    conclusion: str


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Debate Agents (run in parallel)
# -------------------------

def pro_agent(state: State):
    """Argues in favor of the topic."""
    response = llm.invoke(
        f"You are debating: '{state['topic']}'\n\n"
        f"Argue STRONGLY IN FAVOR. Provide 3-4 compelling arguments "
        f"with evidence or reasoning for why this is a good idea."
    )
    return {"pro_arguments": response.content}


def con_agent(state: State):
    """Argues against the topic."""
    response = llm.invoke(
        f"You are debating: '{state['topic']}'\n\n"
        f"Argue STRONGLY AGAINST. Provide 3-4 compelling arguments "
        f"with evidence or reasoning for why this is a bad idea."
    )
    return {"con_arguments": response.content}


# -------------------------
# Moderator
# -------------------------

def moderator(state: State):
    """Summarizes both sides into a balanced conclusion."""
    response = llm.invoke(
        f"You are a neutral moderator. Two debaters argued about: '{state['topic']}'\n\n"
        f"PRO:\n{state['pro_arguments']}\n\n"
        f"CON:\n{state['con_arguments']}\n\n"
        f"Provide a balanced summary with:\n"
        f"1. Strongest point from each side\n"
        f"2. Your neutral conclusion\n"
        f"3. Who should consider which option"
    )
    return {"conclusion": response.content}


# -------------------------
# Build Graph (direct parallel edges)
# -------------------------

builder = StateGraph(State)

builder.add_node("pro_agent", pro_agent)
builder.add_node("con_agent", con_agent)
builder.add_node("moderator", moderator)

# Fan-out: both debate agents run in parallel from START
builder.add_edge(START, "pro_agent")
builder.add_edge(START, "con_agent")

# Fan-in: both flow into moderator
builder.add_edge("pro_agent", "moderator")
builder.add_edge("con_agent", "moderator")
builder.add_edge("moderator", END)

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    topic = input("> Enter a topic to debate: ")

    print("\n⏳ Agents are debating...\n")

    result = graph.invoke({"topic": topic})

    print("🟢 PRO ARGUMENTS:")
    print(result["pro_arguments"])

    print("\n" + "-" * 40 + "\n")

    print("🔴 CON ARGUMENTS:")
    print(result["con_arguments"])

    print("\n" + "=" * 60)
    print("MODERATOR'S CONCLUSION")
    print("=" * 60)
    print(result["conclusion"])
