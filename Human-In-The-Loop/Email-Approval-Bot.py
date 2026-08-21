"""
Pattern: Human-in-the-Loop
Project: HumanApprove - Email Approval Bot

The agent drafts an email, then pauses for human approval before
"sending" it. The human can approve, reject, or ask for edits.

Flow: [Draft] -> [Human reviews] -> [Approve/Edit/Reject] -> [Send or Revise]

Uses LangGraph's conditional edges for the human decision loop.
The "human_review" node collects input interactively (in a real app,
you'd use LangGraph's interrupt() with a checkpointer for persistence).
"""

from typing import TypedDict, Literal

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph


# -------------------------
# State
# -------------------------

class State(TypedDict):
    context: str
    draft: str
    human_feedback: str
    status: str
    final_email: str


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Draft the email
# -------------------------

def draft_email(state: State):
    """AI drafts an email based on context."""
    prompt = f"Draft a professional email based on this context:\n\n{state['context']}"

    if state.get("human_feedback"):
        prompt += f"\n\nPrevious feedback to incorporate:\n{state['human_feedback']}"

    response = llm.invoke(prompt)
    return {"draft": response.content}


# -------------------------
# Human Review (interactive pause)
# -------------------------

def human_review(state: State):
    """Pauses for human to review and provide feedback."""
    print("\n" + "=" * 60)
    print("📧 DRAFT EMAIL FOR REVIEW")
    print("=" * 60)
    print(state["draft"])
    print("=" * 60)

    print("\nOptions:")
    print("  [a] Approve and send")
    print("  [r] Reject (cancel)")
    print("  [e] Edit (provide feedback)")

    choice = input("\nYour choice: ").strip().lower()

    if choice == "a":
        return {"status": "approved", "human_feedback": ""}
    elif choice == "r":
        return {"status": "rejected", "human_feedback": ""}
    else:
        feedback = input("What should be changed? ")
        return {"status": "edit", "human_feedback": feedback}


# -------------------------
# Route based on human decision
# -------------------------

def route_after_review(state: State) -> str:
    if state["status"] == "approved":
        return "send"
    elif state["status"] == "rejected":
        return "cancel"
    return "revise"


# -------------------------
# Terminal nodes
# -------------------------

def send_email(state: State):
    """Simulates sending the approved email."""
    return {"final_email": state["draft"]}


def cancel_email(state: State):
    """Email was rejected by human."""
    return {"final_email": "[CANCELLED] Email was not sent."}


# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(State)

builder.add_node("draft_email", draft_email)
builder.add_node("human_review", human_review)
builder.add_node("send_email", send_email)
builder.add_node("cancel_email", cancel_email)

builder.add_edge(START, "draft_email")
builder.add_edge("draft_email", "human_review")

# Human decision routes to: send, cancel, or loop back to draft
builder.add_conditional_edges(
    "human_review",
    route_after_review,
    {"send": "send_email", "cancel": "cancel_email", "revise": "draft_email"}
)

builder.add_edge("send_email", END)
builder.add_edge("cancel_email", END)

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    print("HumanApprove - Email Approval Bot\n")

    context = input("> What email do you need? (describe the purpose): ")

    result = graph.invoke({"context": context, "human_feedback": "", "status": ""})

    print("\n" + "=" * 60)
    if "[CANCELLED]" in result["final_email"]:
        print("❌ Email cancelled.")
    else:
        print("✅ Email sent!")
        print("=" * 60)
        print(result["final_email"])
