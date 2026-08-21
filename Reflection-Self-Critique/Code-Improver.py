"""
Pattern: Reflection / Self-Critique
Project: SelfCheck AI - Code Improver

A single agent writes code, then reflects on its own output to find
issues and improve it. Unlike Review & Critique, the SAME agent
does both writing and reviewing (self-reflection).

Flow: [Write] -> [Self-reflect] -> [Improve] -> [Final check] -> END

Uses LangGraph's chained builder for a 4-step self-improvement pipeline.
"""

from typing import TypedDict

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph


# -------------------------
# State
# -------------------------

class State(TypedDict):
    task: str
    code: str
    reflection: str
    improved_code: str
    final_check: str


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Nodes
# -------------------------

def write_code(state: State):
    """Writes an initial code solution."""
    response = llm.invoke(
        f"Write a Python function for this task:\n\n{state['task']}\n\n"
        f"Provide working code with basic comments."
    )
    return {"code": response.content}


def self_reflect(state: State):
    """The agent critiques its own code."""
    response = llm.invoke(
        f"You wrote this code:\n\n{state['code']}\n\n"
        f"Now reflect on it critically. Find:\n"
        f"- Bugs or edge cases not handled\n"
        f"- Performance issues\n"
        f"- Readability problems\n"
        f"- Missing error handling\n\n"
        f"Be specific about what needs fixing."
    )
    return {"reflection": response.content}


def improve_code(state: State):
    """Rewrites the code addressing the self-critique."""
    response = llm.invoke(
        f"Original code:\n\n{state['code']}\n\n"
        f"Issues found:\n\n{state['reflection']}\n\n"
        f"Rewrite the code fixing ALL identified issues. "
        f"Make it production-ready with proper error handling."
    )
    return {"improved_code": response.content}


def final_check(state: State):
    """One last check to confirm the improved code is solid."""
    response = llm.invoke(
        f"Review this improved code one final time:\n\n{state['improved_code']}\n\n"
        f"Brief assessment: Is it production-ready? Any remaining concerns? Rate 1-10."
    )
    return {"final_check": response.content}


# -------------------------
# Build Graph (chained builder)
# -------------------------

graph = (
    StateGraph(State)
    .add_node(write_code)
    .add_node(self_reflect)
    .add_node(improve_code)
    .add_node(final_check)
    .add_edge(START, "write_code")
    .add_edge("write_code", "self_reflect")
    .add_edge("self_reflect", "improve_code")
    .add_edge("improve_code", "final_check")
    .add_edge("final_check", END)
    .compile()
)


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    task = input("> Describe the coding task: ")

    print("\n⏳ Writing, reflecting, and improving...\n")

    result = graph.invoke({"task": task})

    print("=" * 60)
    print("INITIAL CODE")
    print("=" * 60)
    print(result["code"])

    print("\n" + "=" * 60)
    print("SELF-REFLECTION")
    print("=" * 60)
    print(result["reflection"])

    print("\n" + "=" * 60)
    print("IMPROVED CODE")
    print("=" * 60)
    print(result["improved_code"])

    print("\n" + "=" * 60)
    print("FINAL CHECK")
    print("=" * 60)
    print(result["final_check"])
