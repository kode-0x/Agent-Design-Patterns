"""
Pattern: Event-Driven Agent
Project: EventBot - Reminder Bot

An event-driven agent that responds to events (user commands).
The agent classifies each event and routes to the appropriate
handler using Command(goto).

Flow: [Event arrives] -> [Classify + Command route] -> [Handler] -> [Wait for next]

Uses LangGraph's Command for combined classification + routing.
"""

import json
from datetime import datetime, timedelta
from typing import TypedDict, Literal

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command


# -------------------------
# In-memory reminder store
# -------------------------

reminders: list[dict] = []


# -------------------------
# State
# -------------------------

class State(TypedDict):
    user_input: str
    event_type: str
    response: str


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Event Classifier (uses Command to route)
# -------------------------

def classify_event(state: State) -> Command[Literal["add", "list", "check", "help"]]:
    """Classifies the user input and routes to the correct handler."""
    response = llm.invoke(
        f"Classify this input into ONE event type: add, list, check, or help.\n\n"
        f"- add: user wants to create a new reminder\n"
        f"- list: user wants to see all reminders\n"
        f"- check: user wants to check what's due\n"
        f"- help: anything else or unclear\n\n"
        f"User: {state['user_input']}\n\n"
        f"Reply with ONLY the event type (one word)."
    )
    event_type = response.content.strip().lower()
    if event_type not in ["add", "list", "check", "help"]:
        event_type = "help"

    return Command(update={"event_type": event_type}, goto=event_type)


# -------------------------
# Event Handlers
# -------------------------

def add(state: State):
    """Handles 'add reminder' events."""
    response = llm.invoke(
        f"Extract reminder details from: {state['user_input']}\n\n"
        f"Return JSON: {{\"task\": \"what to do\", \"minutes\": number_of_minutes}}"
    )

    try:
        data = json.loads(response.content)
        due_time = datetime.now() + timedelta(minutes=data.get("minutes", 60))
        reminder = {"task": data["task"], "due": due_time.strftime("%H:%M")}
        reminders.append(reminder)
        return {"response": f"✅ Reminder added: '{reminder['task']}' - due at {reminder['due']}"}
    except (json.JSONDecodeError, KeyError):
        return {"response": "⚠️ Couldn't parse. Try: 'Remind me to call mom in 30 minutes'"}


def list(state: State):
    """Handles 'list reminders' events."""
    if not reminders:
        return {"response": "📭 No reminders set."}

    lines = ["📋 Your reminders:"]
    for i, r in enumerate(reminders, 1):
        lines.append(f"  {i}. {r['task']} (due: {r['due']})")
    return {"response": "\n".join(lines)}


def check(state: State):
    """Handles 'check due' events."""
    now = datetime.now().strftime("%H:%M")
    due_now = [r for r in reminders if r["due"] <= now]

    if not due_now:
        return {"response": "⏰ Nothing due right now. You're all caught up!"}

    lines = ["🔔 Due NOW:"]
    for r in due_now:
        lines.append(f"  - {r['task']}")
    return {"response": "\n".join(lines)}


def help(state: State):
    """Handles unknown/help events."""
    return {"response": (
        "🤖 EventBot - Reminder Bot\n\n"
        "Commands:\n"
        "  • 'Remind me to [task] in [X] minutes'\n"
        "  • 'Show my reminders' or 'list'\n"
        "  • 'What's due?' or 'check'\n"
        "  • 'exit' to quit"
    )}


# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(State)

builder.add_node("classify_event", classify_event)
builder.add_node("add", add)
builder.add_node("list", list)
builder.add_node("check", check)
builder.add_node("help", help)

builder.add_edge(START, "classify_event")
builder.add_edge("add", END)
builder.add_edge("list", END)
builder.add_edge("check", END)
builder.add_edge("help", END)

graph = builder.compile()


# -------------------------
# Event Loop
# -------------------------

if __name__ == "__main__":
    print("🔔 EventBot - Reminder Bot")
    print("Type 'help' for commands, 'exit' to quit.\n")

    while True:
        user_input = input("> ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        result = graph.invoke({"user_input": user_input})
        print(f"\n{result['response']}\n")
