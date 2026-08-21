"""
Pattern: Handoff
Project: SupportFlow - Customer Support Bot

The handoff pattern passes conversation between specialized agents.
A triage node determines the issue type, then hands off to the
appropriate specialist using Command(goto).

Flow: [Triage + handoff via Command] -> [Billing | Technical | General] specialist

Uses LangGraph's Command for state update + routing in one step.
"""

from typing import TypedDict, Literal

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command


# -------------------------
# State
# -------------------------

class State(TypedDict):
    customer_message: str
    department: str
    response: str


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Triage: Classify and handoff
# -------------------------

def triage(state: State) -> Command[Literal["billing", "technical", "general"]]:
    """Determines which department should handle the customer and hands off."""
    response = llm.invoke(
        f"You are a customer support triage agent. "
        f"Determine which department should handle this message.\n\n"
        f"Customer: {state['customer_message']}\n\n"
        f"Reply with ONLY one word: billing, technical, or general"
    )
    department = response.content.strip().lower()

    if department not in ["billing", "technical", "general"]:
        department = "general"

    return Command(update={"department": department}, goto=department)


# -------------------------
# Specialist Nodes
# -------------------------

def billing(state: State):
    """Handles billing, payments, and subscription issues."""
    response = llm.invoke(
        f"You are a billing specialist. Help this customer with their billing issue.\n"
        f"Be professional, empathetic, and solution-oriented.\n\n"
        f"Customer: {state['customer_message']}"
    )
    return {"response": response.content}


def technical(state: State):
    """Handles technical issues, bugs, and how-to questions."""
    response = llm.invoke(
        f"You are a technical support specialist. Help this customer.\n"
        f"Provide clear step-by-step solutions.\n\n"
        f"Customer: {state['customer_message']}"
    )
    return {"response": response.content}


def general(state: State):
    """Handles general inquiries, feedback, and other questions."""
    response = llm.invoke(
        f"You are a general support agent. Help this customer.\n"
        f"Be friendly and helpful.\n\n"
        f"Customer: {state['customer_message']}"
    )
    return {"response": response.content}


# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(State)

builder.add_node("triage", triage)
builder.add_node("billing", billing)
builder.add_node("technical", technical)
builder.add_node("general", general)

builder.add_edge(START, "triage")
builder.add_edge("billing", END)
builder.add_edge("technical", END)
builder.add_edge("general", END)

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    print("SupportFlow - Customer Support Bot")
    print("Type your message (or 'exit' to quit):\n")

    while True:
        message = input("Customer> ")

        if message.lower() == "exit":
            print("Thank you for contacting support!")
            break

        result = graph.invoke({"customer_message": message})

        print(f"\n[Handed off to: {result['department'].upper()}]")
        print(f"Agent: {result['response']}\n")
