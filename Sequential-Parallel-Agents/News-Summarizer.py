"""
Pattern: Sequential + Parallel Agents
Project: NewsFlow - News Summarizer

Combines both patterns:
1. Parallel: Multiple nodes search for news simultaneously (fan-out from START)
2. Sequential: Results flow into summarize -> format (chained)

Flow: START -> [breaking | analysis | background] -> summarize -> format -> END
"""

from typing import TypedDict

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph
from tavily import TavilyClient


# -------------------------
# State
# -------------------------

class State(TypedDict):
    topic: str
    breaking_news: str
    analysis: str
    background: str
    summary: str
    report: str


# -------------------------
# LLM & Tools
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")
tavily = TavilyClient()


# -------------------------
# Parallel: News Search Nodes
# -------------------------

def search_breaking(state: State):
    """Searches for breaking/latest news."""
    result = tavily.search(query=f"{state['topic']} breaking news today", max_results=3)
    content = "\n".join([r["content"] for r in result["results"]])
    return {"breaking_news": content}


def search_analysis(state: State):
    """Searches for expert analysis and opinions."""
    result = tavily.search(query=f"{state['topic']} expert analysis opinion", max_results=3)
    content = "\n".join([r["content"] for r in result["results"]])
    return {"analysis": content}


def search_background(state: State):
    """Searches for background context on the topic."""
    result = tavily.search(query=f"{state['topic']} background context explained", max_results=3)
    content = "\n".join([r["content"] for r in result["results"]])
    return {"background": content}


# -------------------------
# Sequential: Summarize -> Format
# -------------------------

def summarize(state: State):
    """Summarizes all parallel results into key points."""
    all_news = (
        f"BREAKING:\n{state['breaking_news']}\n\n"
        f"ANALYSIS:\n{state['analysis']}\n\n"
        f"BACKGROUND:\n{state['background']}"
    )
    response = llm.invoke(
        f"Summarize this news about '{state['topic']}' into 5-7 key bullet points:\n\n{all_news}"
    )
    return {"summary": response.content}


def format_report(state: State):
    """Formats the summary into a polished news report."""
    response = llm.invoke(
        f"Turn these bullet points into a short, well-structured news report "
        f"with a headline and 2-3 paragraphs:\n\n{state['summary']}"
    )
    return {"report": response.content}


# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(State)

builder.add_node("search_breaking", search_breaking)
builder.add_node("search_analysis", search_analysis)
builder.add_node("search_background", search_background)
builder.add_node("summarize", summarize)
builder.add_node("format_report", format_report)

# Parallel fan-out from START
builder.add_edge(START, "search_breaking")
builder.add_edge(START, "search_analysis")
builder.add_edge(START, "search_background")

# Fan-in to sequential processing
builder.add_edge("search_breaking", "summarize")
builder.add_edge("search_analysis", "summarize")
builder.add_edge("search_background", "summarize")

# Sequential chain
builder.add_edge("summarize", "format_report")
builder.add_edge("format_report", END)

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    topic = input("> Enter a news topic: ")

    print("\n⏳ Gathering and summarizing news...\n")

    result = graph.invoke({"topic": topic})

    print("=" * 60)
    print("NEWS REPORT")
    print("=" * 60)
    print(result["report"])
