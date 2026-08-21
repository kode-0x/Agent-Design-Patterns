"""
Pattern: Review & Critique Loop
Project: ImproveLoop - Essay Improver

Like Review & Critique, but the writer revises based on feedback
in a loop until the critic approves (or max iterations reached).

Flow: [Write] -> [Critique] -> (needs revision?) -> [Revise] -> [Critique] -> ... -> END

Uses LangGraph's conditional edges for the revision loop.
"""

from typing import TypedDict

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph


# -------------------------
# Config
# -------------------------

MAX_REVISIONS = 3


# -------------------------
# State
# -------------------------

class State(TypedDict):
    topic: str
    essay: str
    critique: str
    revision_count: int
    is_approved: bool


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Nodes
# -------------------------

def writer(state: State):
    """Writes the initial essay."""
    response = llm.invoke(
        f"Write a short essay (about 300 words) on: '{state['topic']}'\n\n"
        f"Make it well-structured with clear arguments."
    )
    return {"essay": response.content, "revision_count": 0}


def critic(state: State):
    """Reviews the essay and decides if it passes."""
    response = llm.invoke(
        f"You are a strict essay critic. Review this essay.\n\n"
        f"Essay:\n{state['essay']}\n\n"
        f"Provide 2-3 specific improvements needed.\n"
        f"On the LAST line, write exactly APPROVED or NEEDS_REVISION."
    )

    content = response.content.strip()
    is_approved = "APPROVED" in content.split("\n")[-1].upper()

    return {"critique": content, "is_approved": is_approved}


def reviser(state: State):
    """Revises the essay based on the critic's feedback."""
    response = llm.invoke(
        f"Revise this essay based on the feedback below.\n\n"
        f"Original essay:\n{state['essay']}\n\n"
        f"Feedback:\n{state['critique']}\n\n"
        f"Write the improved version (keep it around 300 words)."
    )
    return {"essay": response.content, "revision_count": state["revision_count"] + 1}


# -------------------------
# Conditional: revise or finish
# -------------------------

def should_revise(state: State) -> str:
    """Loop back to reviser or finish."""
    if state["is_approved"]:
        return "done"
    if state["revision_count"] >= MAX_REVISIONS:
        return "done"
    return "revise"


# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(State)

builder.add_node("writer", writer)
builder.add_node("critic", critic)
builder.add_node("reviser", reviser)

builder.add_edge(START, "writer")
builder.add_edge("writer", "critic")

builder.add_conditional_edges(
    "critic", should_revise, {"revise": "reviser", "done": END}
)

# Revision loop: reviser -> critic -> (check again)
builder.add_edge("reviser", "critic")

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    topic = input("> Enter an essay topic: ")

    print(f"\n⏳ Writing and improving (up to {MAX_REVISIONS} revisions)...\n")

    result = graph.invoke({"topic": topic, "revision_count": 0, "is_approved": False})

    print("=" * 60)
    print(f"FINAL ESSAY (after {result['revision_count']} revision(s))")
    print("=" * 60)
    print(result["essay"])

    print("\n" + "=" * 60)
    print("FINAL CRITIQUE")
    print("=" * 60)
    print(result["critique"])

    if result["is_approved"]:
        print("\n✅ Essay APPROVED by critic!")
    else:
        print(f"\n⚠️ Max revisions ({MAX_REVISIONS}) reached.")
