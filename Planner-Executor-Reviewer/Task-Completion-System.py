"""
Pattern: Planner-Executor-Reviewer
Project: PlanBuildReview - Task Completion System

Three-stage pattern: a planner creates a plan, an executor carries it out,
and a reviewer checks the quality. If rejected, loops back to executor.

Flow: [Plan] -> [Execute] -> [Review] -> (rejected?) -> [Execute] -> [Review] -> ... -> END

Uses LangGraph's conditional edges for the review loop + SQLite for history.
"""

import sqlite3
from typing import TypedDict

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph


# -------------------------
# Config
# -------------------------

MAX_ATTEMPTS = 3


# -------------------------
# Database
# -------------------------

def init_db():
    conn = sqlite3.connect("plan_build_review.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def log_to_db(stage: str, content: str):
    conn = init_db()
    conn.execute("INSERT INTO history (stage, content) VALUES (?, ?)", (stage, content))
    conn.commit()
    conn.close()


# -------------------------
# State
# -------------------------

class State(TypedDict):
    task: str
    plan: str
    execution: str
    review: str
    is_approved: bool
    attempts: int


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Planner
# -------------------------

def planner(state: State):
    """Creates a detailed plan for the task."""
    response = llm.invoke(
        f"Create a clear, step-by-step plan for this task:\n\n"
        f"Task: {state['task']}\n\n"
        f"Provide 3-5 specific, actionable steps."
    )
    log_to_db("plan", response.content)
    return {"plan": response.content}


# -------------------------
# Executor
# -------------------------

def executor(state: State):
    """Executes the plan and produces a result."""
    feedback = ""
    if state.get("review") and not state.get("is_approved"):
        feedback = f"\n\nPrevious review feedback to address:\n{state['review']}"

    response = llm.invoke(
        f"Follow this plan and produce the result:\n\n"
        f"Plan:\n{state['plan']}\n"
        f"Task: {state['task']}{feedback}\n\n"
        f"Provide a complete, detailed implementation."
    )
    log_to_db("execution", response.content)
    return {"execution": response.content, "attempts": state.get("attempts", 0) + 1}


# -------------------------
# Reviewer
# -------------------------

def reviewer(state: State):
    """Reviews the execution quality."""
    response = llm.invoke(
        f"You are a strict quality reviewer. Review this execution:\n\n"
        f"Original task: {state['task']}\n"
        f"Plan: {state['plan']}\n"
        f"Execution:\n{state['execution']}\n\n"
        f"Evaluate completeness, correctness, and quality.\n"
        f"On the LAST line, write exactly: APPROVED or NEEDS_REVISION"
    )

    content = response.content.strip()
    is_approved = "APPROVED" in content.split("\n")[-1].upper()

    log_to_db("review", content)
    return {"review": content, "is_approved": is_approved}


# -------------------------
# Conditional: loop or finish
# -------------------------

def should_retry(state: State) -> str:
    """Loop back to executor if rejected, otherwise end."""
    if state["is_approved"]:
        return "done"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "done"
    return "retry"


# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(State)

builder.add_node("planner", planner)
builder.add_node("executor", executor)
builder.add_node("reviewer", reviewer)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "reviewer")

# Review loop: retry goes back to executor, done exits
builder.add_conditional_edges(
    "reviewer", should_retry, {"retry": "executor", "done": END}
)

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    print("PlanBuildReview - Task Completion System\n")

    task = input("> Describe your task: ")

    print(f"\n⏳ Planning, executing, and reviewing (up to {MAX_ATTEMPTS} attempts)...\n")

    result = graph.invoke({"task": task, "attempts": 0, "is_approved": False})

    print("=" * 60)
    print("PLAN")
    print("=" * 60)
    print(result["plan"])

    print("\n" + "=" * 60)
    print(f"EXECUTION (attempt {result['attempts']})")
    print("=" * 60)
    print(result["execution"])

    print("\n" + "=" * 60)
    print("REVIEW")
    print("=" * 60)
    print(result["review"])

    if result["is_approved"]:
        print("\n✅ Task APPROVED by reviewer!")
    else:
        print(f"\n⚠️ Max attempts ({MAX_ATTEMPTS}) reached.")
