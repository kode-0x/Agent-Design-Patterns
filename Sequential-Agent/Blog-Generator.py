"""
Pattern: Sequential Agent
Project: BlogFlow - A blog generator with sequential steps

Sequential pattern: each node runs one after another in a chain.
Step 1: Research the topic
Step 2: Create an outline
Step 3: Write the full blog post

Uses LangGraph's chained builder pattern (.add_node().add_edge()).
"""

from typing import TypedDict

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph


# -------------------------
# State
# -------------------------

class State(TypedDict):
    topic: str
    research: str
    outline: str
    blog_post: str


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Nodes
# -------------------------

def research(state: State):
    """Step 1: Research the topic."""
    response = llm.invoke(
        f"Research the topic '{state['topic']}'. "
        f"Provide key facts, statistics, and talking points. "
        f"Keep it concise - bullet points are fine."
    )
    return {"research": response.content}


def outline(state: State):
    """Step 2: Create an outline from the research."""
    response = llm.invoke(
        f"Based on this research:\n\n{state['research']}\n\n"
        f"Create a blog post outline for '{state['topic']}'. "
        f"Include an intro, 3-4 main sections, and a conclusion."
    )
    return {"outline": response.content}


def write(state: State):
    """Step 3: Write the full blog post."""
    response = llm.invoke(
        f"Write a complete blog post based on this outline:\n\n{state['outline']}\n\n"
        f"Use this research:\n\n{state['research']}\n\n"
        f"Make it engaging, informative, and around 500 words."
    )
    return {"blog_post": response.content}


# -------------------------
# Build Graph (chained builder pattern)
# -------------------------

graph = (
    StateGraph(State)
    .add_node(research)
    .add_node(outline)
    .add_node(write)
    .add_edge(START, "research")
    .add_edge("research", "outline")
    .add_edge("outline", "write")
    .add_edge("write", END)
    .compile()
)


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    topic = input("> Enter a topic for your blog post: ")

    print("\n⏳ Generating blog post...\n")

    result = graph.invoke({"topic": topic})

    print("=" * 60)
    print("RESEARCH")
    print("=" * 60)
    print(result["research"])

    print("\n" + "=" * 60)
    print("OUTLINE")
    print("=" * 60)
    print(result["outline"])

    print("\n" + "=" * 60)
    print("FINAL BLOG POST")
    print("=" * 60)
    print(result["blog_post"])
