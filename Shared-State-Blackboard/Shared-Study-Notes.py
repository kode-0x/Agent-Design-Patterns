"""
Pattern: Shared State / Blackboard
Project: SharedBrain - Shared Study Notes

Multiple specialist agents write to a shared "blackboard" (SQLite database)
in parallel. Each agent contributes its expertise, then a synthesizer
reads ALL contributions from the shared DB to create a unified study guide.

Flow: START -> [Concepts | Examples | Quiz] (parallel) -> [Synthesizer reads DB] -> END

Uses LangGraph's direct parallel edges + SQLite as the shared blackboard.
"""

import sqlite3
from typing import TypedDict

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph


# -------------------------
# Shared Blackboard (SQLite)
# -------------------------

def init_blackboard():
    conn = sqlite3.connect("blackboard.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            agent_role TEXT,
            content TEXT
        )
    """)
    conn.commit()
    return conn


def write_to_blackboard(topic: str, role: str, content: str):
    """Write a note to the shared blackboard."""
    conn = init_blackboard()
    conn.execute(
        "INSERT INTO notes (topic, agent_role, content) VALUES (?, ?, ?)",
        (topic, role, content)
    )
    conn.commit()
    conn.close()


def read_blackboard(topic: str) -> list[dict]:
    """Read all notes for a topic from the blackboard."""
    conn = init_blackboard()
    cursor = conn.execute(
        "SELECT agent_role, content FROM notes WHERE topic = ?", (topic,)
    )
    notes = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    conn.close()
    return notes


# -------------------------
# State
# -------------------------

class State(TypedDict):
    topic: str
    concepts: str
    examples: str
    quiz: str
    study_guide: str


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Specialist Agents (write to blackboard in parallel)
# -------------------------

def concept_agent(state: State):
    """Explains core concepts and definitions."""
    response = llm.invoke(
        f"For the topic '{state['topic']}', provide clear definitions "
        f"and core concepts (3-5 key points)."
    )
    write_to_blackboard(state["topic"], "concepts", response.content)
    return {"concepts": response.content}


def examples_agent(state: State):
    """Provides practical examples."""
    response = llm.invoke(
        f"For the topic '{state['topic']}', provide 2-3 practical, "
        f"easy-to-understand examples."
    )
    write_to_blackboard(state["topic"], "examples", response.content)
    return {"examples": response.content}


def quiz_agent(state: State):
    """Creates quiz questions for self-testing."""
    response = llm.invoke(
        f"For the topic '{state['topic']}', create 3 quiz questions "
        f"(with answers) to test understanding."
    )
    write_to_blackboard(state["topic"], "quiz", response.content)
    return {"quiz": response.content}


# -------------------------
# Synthesizer (reads from shared blackboard)
# -------------------------

def synthesizer(state: State):
    """Reads all notes from the blackboard and creates a study guide."""
    notes = read_blackboard(state["topic"])
    all_notes = "\n\n".join([f"[{n['role'].upper()}]:\n{n['content']}" for n in notes])

    response = llm.invoke(
        f"Using these shared notes about '{state['topic']}', "
        f"create a concise study guide:\n\n{all_notes}\n\n"
        f"Structure: Overview, Key Concepts, Examples, Self-Test Questions."
    )
    return {"study_guide": response.content}


# -------------------------
# Build Graph (direct parallel edges)
# -------------------------

builder = StateGraph(State)

builder.add_node("concept_agent", concept_agent)
builder.add_node("examples_agent", examples_agent)
builder.add_node("quiz_agent", quiz_agent)
builder.add_node("synthesizer", synthesizer)

# Fan-out: specialists run in parallel
builder.add_edge(START, "concept_agent")
builder.add_edge(START, "examples_agent")
builder.add_edge(START, "quiz_agent")

# Fan-in: all write to blackboard, then synthesizer reads it
builder.add_edge("concept_agent", "synthesizer")
builder.add_edge("examples_agent", "synthesizer")
builder.add_edge("quiz_agent", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    print("SharedBrain - Shared Study Notes\n")

    topic = input("> What topic would you like to study? ")

    print("\n⏳ Specialists are writing to the blackboard...\n")

    result = graph.invoke({"topic": topic})

    print("=" * 60)
    print("STUDY GUIDE")
    print("=" * 60)
    print(result["study_guide"])
