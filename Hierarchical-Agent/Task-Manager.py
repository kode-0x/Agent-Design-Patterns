"""
Pattern: Hierarchical Agent
Project: TaskTree AI - Simple Task Manager

A manager agent delegates work to team lead agents in a hierarchy.
The manager breaks the project into team tasks, then dispatches
team leads dynamically using the Send API.

Flow: [Manager plans] -> Send() -> [Team Leads in parallel] -> [Assemble report]

Uses LangGraph's Send API for dynamic fan-out (number of teams
determined at runtime), plus SQLite for persistence.
"""

import sqlite3
import json
from typing import TypedDict, Annotated
import operator

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send


# -------------------------
# Database
# -------------------------

def init_db():
    conn = sqlite3.connect("task_tree.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            description TEXT,
            result TEXT
        )
    """)
    conn.commit()
    return conn


# -------------------------
# State
# -------------------------

class State(TypedDict):
    project: str
    team_tasks: list[str]
    completed_sections: Annotated[list[str], operator.add]
    final_report: str


# Worker state (single team lead)
class TeamLeadState(TypedDict):
    task: str
    completed_sections: Annotated[list[str], operator.add]


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Manager: Top of hierarchy
# -------------------------

def manager(state: State):
    """Manager breaks the project into team-level tasks."""
    response = llm.invoke(
        f"You are a project manager. Break this project into 2-3 team tasks.\n\n"
        f"Project: {state['project']}\n\n"
        f"Return ONLY a JSON array of task descriptions.\n"
        f'Example: ["Design the database", "Build the API", "Create the UI"]'
    )

    try:
        tasks = json.loads(response.content)
    except json.JSONDecodeError:
        tasks = [state["project"]]

    # Log to DB
    conn = init_db()
    conn.execute(
        "INSERT INTO tasks (role, description) VALUES (?, ?)",
        ("manager", state["project"])
    )
    conn.commit()
    conn.close()

    return {"team_tasks": tasks}


# -------------------------
# Team Lead: Executes a delegated task
# -------------------------

def team_lead(state: TeamLeadState):
    """Team lead produces a detailed solution for their task."""
    response = llm.invoke(
        f"You are a team lead. Execute this task with a detailed plan "
        f"and implementation notes:\n\nTask: {state['task']}"
    )

    # Log to DB
    conn = init_db()
    conn.execute(
        "INSERT INTO tasks (role, description, result) VALUES (?, ?, ?)",
        ("team_lead", state["task"], response.content)
    )
    conn.commit()
    conn.close()

    return {"completed_sections": [f"📋 {state['task']}:\n{response.content}"]}


# -------------------------
# Manager: Assemble final report
# -------------------------

def assemble_report(state: State):
    """Manager assembles all team results into a final report."""
    all_results = "\n\n---\n\n".join(state["completed_sections"])

    response = llm.invoke(
        f"You are a project manager. Your teams completed their tasks.\n\n"
        f"Project: {state['project']}\n\n"
        f"Team results:\n{all_results}\n\n"
        f"Write a brief project completion report."
    )
    return {"final_report": response.content}


# -------------------------
# Dynamic dispatch with Send API
# -------------------------

def assign_teams(state: State):
    """Dispatches a team lead for each task using Send API."""
    return [Send("team_lead", {"task": task}) for task in state["team_tasks"]]


# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(State)

builder.add_node("manager", manager)
builder.add_node("team_lead", team_lead)
builder.add_node("assemble_report", assemble_report)

builder.add_edge(START, "manager")
builder.add_conditional_edges("manager", assign_teams, ["team_lead"])
builder.add_edge("team_lead", "assemble_report")
builder.add_edge("assemble_report", END)

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    print("TaskTree AI - Hierarchical Task Manager\n")

    project = input("> Describe your project: ")

    print("\n⏳ Manager is delegating to teams...\n")

    result = graph.invoke({"project": project, "team_tasks": [], "completed_sections": []})

    print("=" * 60)
    print("PROJECT REPORT")
    print("=" * 60)
    print(result["final_report"])
