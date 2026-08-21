"""
Pattern: Planner-Executor
Project: PlanDo AI - To-Do List Planner

The planner creates a step-by-step plan, then the executor
carries out each step in a loop. Uses SQLite to persist the to-do list.

Flow: [Planner creates steps] -> [Executor loop] -> [Store in DB] -> [Summarize]

Uses LangGraph's conditional edges for looping until all steps are done.
"""

import sqlite3
import json
from typing import TypedDict

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph


# -------------------------
# Database Setup
# -------------------------

def init_db():
    conn = sqlite3.connect("todos.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            details TEXT DEFAULT ''
        )
    """)
    conn.commit()
    return conn


# -------------------------
# State
# -------------------------

class State(TypedDict):
    goal: str
    plan: list[str]
    current_step: int
    execution_log: list[str]
    final_summary: str


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Planner Node
# -------------------------

def planner(state: State):
    """Creates a step-by-step plan to achieve the goal."""
    response = llm.invoke(
        f"Break this goal into 3-5 clear, actionable to-do items.\n\n"
        f"Goal: {state['goal']}\n\n"
        f"Return ONLY a JSON array of strings.\n"
        f'Example: ["Research topic", "Write first draft", "Review and edit"]'
    )

    try:
        plan = json.loads(response.content)
    except json.JSONDecodeError:
        plan = [state["goal"]]

    return {"plan": plan, "current_step": 0, "execution_log": []}


# -------------------------
# Executor Node
# -------------------------

def executor(state: State):
    """Executes the current step and stores it in the database."""
    step = state["plan"][state["current_step"]]

    response = llm.invoke(
        f"You are executing this task: '{step}'\n"
        f"(Part of the goal: '{state['goal']}')\n\n"
        f"Provide a brief description of what was done (1-2 sentences)."
    )

    # Store in database
    conn = init_db()
    conn.execute(
        "INSERT INTO todos (task, status, details) VALUES (?, ?, ?)",
        (step, "completed", response.content)
    )
    conn.commit()
    conn.close()

    log = state["execution_log"] + [f"✅ {step}: {response.content}"]
    return {"current_step": state["current_step"] + 1, "execution_log": log}


# -------------------------
# Conditional: more steps?
# -------------------------

def should_continue(state: State) -> str:
    """Loop back to executor if steps remain, otherwise summarize."""
    if state["current_step"] < len(state["plan"]):
        return "executor"
    return "summarize"


# -------------------------
# Summarize Node
# -------------------------

def summarize(state: State):
    """Summarizes what was accomplished."""
    log_text = "\n".join(state["execution_log"])
    return {"final_summary": f"Completed {len(state['plan'])} tasks:\n\n{log_text}"}


# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(State)

builder.add_node("planner", planner)
builder.add_node("executor", executor)
builder.add_node("summarize", summarize)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")

# Loop: executor -> check -> executor OR summarize
builder.add_conditional_edges(
    "executor", should_continue, {"executor": "executor", "summarize": "summarize"}
)

builder.add_edge("summarize", END)

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    print("PlanDo AI - To-Do List Planner")
    goal = input("\n> What's your goal? ")

    print("\n⏳ Planning and executing...\n")

    result = graph.invoke({"goal": goal})

    print("=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(result["final_summary"])
