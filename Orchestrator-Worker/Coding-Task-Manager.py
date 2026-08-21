"""
Pattern: Orchestrator-Worker
Project: TaskMaster AI - Coding Task Manager

The orchestrator breaks a coding task into subtasks, then uses the
Send API to dynamically dispatch workers for each subtask.
Workers execute in parallel and report back to a synthesizer.

Flow: [Orchestrator plans] -> Send() -> [Workers in parallel] -> [Synthesizer assembles]

Uses LangGraph's Send API for dynamic fan-out (number of workers
determined at runtime by the orchestrator).
"""

from typing import TypedDict, Annotated
import operator
import json

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send


# -------------------------
# State
# -------------------------

class State(TypedDict):
    task: str
    subtasks: list[str]
    completed_sections: Annotated[list[str], operator.add]
    final_output: str


# Worker state (receives a single subtask)
class WorkerState(TypedDict):
    subtask: str
    completed_sections: Annotated[list[str], operator.add]


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Orchestrator: Break down the task
# -------------------------

def orchestrator(state: State):
    """Breaks a coding task into subtasks."""
    response = llm.invoke(
        f"Break this coding task into 2-4 small, independent subtasks.\n\n"
        f"Task: {state['task']}\n\n"
        f"Return ONLY a JSON array of strings.\n"
        f'Example: ["Create the data model", "Write the API handler", "Add validation"]'
    )

    try:
        subtasks = json.loads(response.content)
    except json.JSONDecodeError:
        subtasks = [state["task"]]

    return {"subtasks": subtasks}


# -------------------------
# Worker: Execute a subtask
# -------------------------

def worker(state: WorkerState):
    """A worker that writes code for a single subtask."""
    response = llm.invoke(
        f"You are a coding worker. Complete this subtask with clean, "
        f"well-commented code:\n\nSubtask: {state['subtask']}"
    )
    return {"completed_sections": [f"## {state['subtask']}\n\n{response.content}"]}


# -------------------------
# Synthesizer: Assemble results
# -------------------------

def synthesizer(state: State):
    """Combines all worker results into a final cohesive solution."""
    completed = "\n\n---\n\n".join(state["completed_sections"])

    response = llm.invoke(
        f"Combine these code solutions into one cohesive, working implementation:\n\n"
        f"{completed}\n\nProvide the final combined code with explanations."
    )
    return {"final_output": response.content}


# -------------------------
# Dynamic fan-out with Send API
# -------------------------

def assign_workers(state: State):
    """Dispatches a worker for each subtask using Send API."""
    return [Send("worker", {"subtask": s}) for s in state["subtasks"]]


# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(State)

builder.add_node("orchestrator", orchestrator)
builder.add_node("worker", worker)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", assign_workers, ["worker"])
builder.add_edge("worker", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    task = input("> Describe your coding task: ")

    print("\n⏳ Orchestrator is breaking down the task...\n")

    result = graph.invoke({"task": task, "subtasks": [], "completed_sections": []})

    print("=" * 60)
    print("FINAL SOLUTION")
    print("=" * 60)
    print(result["final_output"])
