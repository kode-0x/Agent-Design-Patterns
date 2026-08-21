"""
Pattern: Review & Critique
Project: CritiqueAI - Essay Checker

A writer node produces an essay, then a critic node reviews it
and provides feedback. This is a single-pass review (no loop).

Flow: [Writer] -> [Critic] -> END

Uses LangGraph's chained builder pattern for a simple two-step pipeline.
"""

from typing import TypedDict

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph


# -------------------------
# State
# -------------------------

class State(TypedDict):
    topic: str
    essay: str
    critique: str


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Nodes
# -------------------------

def writer(state: State):
    """Writes an essay on the given topic."""
    response = llm.invoke(
        f"Write a short essay (about 300 words) on: '{state['topic']}'\n\n"
        f"Make it well-structured with an introduction, body, and conclusion."
    )
    return {"essay": response.content}


def critic(state: State):
    """Reviews the essay and provides structured feedback."""
    response = llm.invoke(
        f"You are an essay critic. Review this essay and provide feedback.\n\n"
        f"Essay:\n{state['essay']}\n\n"
        f"Provide your critique in this format:\n"
        f"- Strengths (2-3 points)\n"
        f"- Weaknesses (2-3 points)\n"
        f"- Suggestions for improvement\n"
        f"- Overall score (1-10)"
    )
    return {"critique": response.content}


# -------------------------
# Build Graph (chained builder)
# -------------------------

graph = (
    StateGraph(State)
    .add_node(writer)
    .add_node(critic)
    .add_edge(START, "writer")
    .add_edge("writer", "critic")
    .add_edge("critic", END)
    .compile()
)


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    topic = input("> Enter an essay topic: ")

    print("\n⏳ Writing and reviewing essay...\n")

    result = graph.invoke({"topic": topic})

    print("=" * 60)
    print("ESSAY")
    print("=" * 60)
    print(result["essay"])

    print("\n" + "=" * 60)
    print("CRITIQUE")
    print("=" * 60)
    print(result["critique"])
