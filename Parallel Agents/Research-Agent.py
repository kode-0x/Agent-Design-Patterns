"""
Pattern: Parallel Agents
Project: ResearchHub - Parallel research on a topic

Parallel pattern: multiple nodes run simultaneously from START,
then results flow into a single aggregator node.

Uses LangGraph's direct parallel edges (START -> multiple nodes).
No router needed - the graph handles fan-out automatically.
"""

from typing import TypedDict

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph
from tavily import TavilyClient


# -------------------------
# State
# -------------------------

class State(TypedDict):
    query: str
    general_results: str
    technical_results: str
    latest_results: str
    summary: str


# -------------------------
# LLM & Tools
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")
tavily = TavilyClient()


# -------------------------
# Parallel Research Nodes
# -------------------------

def general_research(state: State):
    """Searches for general information on the topic."""
    result = tavily.search(query=state["query"], max_results=3)
    content = "\n".join([r["content"] for r in result["results"]])
    return {"general_results": content}


def technical_research(state: State):
    """Searches for technical details on the topic."""
    result = tavily.search(query=f"{state['query']} technical details", max_results=3)
    content = "\n".join([r["content"] for r in result["results"]])
    return {"technical_results": content}


def latest_research(state: State):
    """Searches for the latest news on the topic."""
    result = tavily.search(query=f"{state['query']} latest news", max_results=3)
    content = "\n".join([r["content"] for r in result["results"]])
    return {"latest_results": content}


# -------------------------
# Aggregator
# -------------------------

def summarize(state: State):
    """Combines all parallel research into a cohesive summary."""
    all_research = (
        f"GENERAL:\n{state['general_results']}\n\n"
        f"TECHNICAL:\n{state['technical_results']}\n\n"
        f"LATEST:\n{state['latest_results']}"
    )

    response = llm.invoke(
        f"Summarize this research about '{state['query']}' "
        f"into a clear, organized report:\n\n{all_research}"
    )
    return {"summary": response.content}


# -------------------------
# Build Graph (direct parallel edges from START)
# -------------------------

builder = StateGraph(State)

builder.add_node("general_research", general_research)
builder.add_node("technical_research", technical_research)
builder.add_node("latest_research", latest_research)
builder.add_node("summarize", summarize)

# Fan-out: START connects to all three nodes in parallel
builder.add_edge(START, "general_research")
builder.add_edge(START, "technical_research")
builder.add_edge(START, "latest_research")

# Fan-in: all three nodes connect to summarize
builder.add_edge("general_research", "summarize")
builder.add_edge("technical_research", "summarize")
builder.add_edge("latest_research", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    query = input("> Enter a research topic: ")

    print("\n⏳ Researching in parallel...\n")

    result = graph.invoke({"query": query})

    print("=" * 60)
    print("RESEARCH SUMMARY")
    print("=" * 60)
    print(result["summary"])
