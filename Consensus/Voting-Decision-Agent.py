"""
Pattern: Consensus
Project: ConsensusAI - Voting/Decision Agent

Multiple agents independently evaluate a decision in parallel,
then a tally node counts votes and announces the consensus.
Each agent has a different perspective/persona.

Flow: START -> [Practical | Creative | Cautious] (parallel) -> [Tally] -> END

Uses LangGraph's direct parallel edges for simultaneous voting.
"""

from typing import TypedDict, Annotated
import operator

from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph


# -------------------------
# State
# -------------------------

class State(TypedDict):
    question: str
    options: list[str]
    practical_vote: str
    creative_vote: str
    cautious_vote: str
    decision: str


# -------------------------
# LLM
# -------------------------

llm = ChatOpenRouter(model="anthropic/claude-sonnet-4")


# -------------------------
# Voting Agents (run in parallel, different perspectives)
# -------------------------

def practical_agent(state: State):
    """Votes based on practicality and feasibility."""
    options_str = "\n".join([f"- {opt}" for opt in state["options"]])
    response = llm.invoke(
        f"You are a PRACTICAL thinker. Consider feasibility, cost, and effort.\n\n"
        f"Question: {state['question']}\nOptions:\n{options_str}\n\n"
        f"Vote for ONE option. Reply with ONLY the option text you choose."
    )
    return {"practical_vote": response.content.strip()}


def creative_agent(state: State):
    """Votes based on creativity and innovation."""
    options_str = "\n".join([f"- {opt}" for opt in state["options"]])
    response = llm.invoke(
        f"You are a CREATIVE thinker. Consider innovation, uniqueness, and excitement.\n\n"
        f"Question: {state['question']}\nOptions:\n{options_str}\n\n"
        f"Vote for ONE option. Reply with ONLY the option text you choose."
    )
    return {"creative_vote": response.content.strip()}


def cautious_agent(state: State):
    """Votes based on risk assessment and safety."""
    options_str = "\n".join([f"- {opt}" for opt in state["options"]])
    response = llm.invoke(
        f"You are a CAUTIOUS thinker. Consider risks, downsides, and safety.\n\n"
        f"Question: {state['question']}\nOptions:\n{options_str}\n\n"
        f"Vote for ONE option. Reply with ONLY the option text you choose."
    )
    return {"cautious_vote": response.content.strip()}


# -------------------------
# Tally votes and decide
# -------------------------

def tally_votes(state: State):
    """Counts votes and announces the consensus decision."""
    votes = (
        f"Practical Agent voted: {state['practical_vote']}\n"
        f"Creative Agent voted: {state['creative_vote']}\n"
        f"Cautious Agent voted: {state['cautious_vote']}"
    )

    response = llm.invoke(
        f"You are a neutral judge. Here are the votes from 3 agents:\n\n"
        f"{votes}\n\n"
        f"The original options were: {state['options']}\n\n"
        f"Determine the consensus (majority wins). Announce:\n"
        f"1. How each agent voted and why that fits their perspective\n"
        f"2. The winning option\n"
        f"3. Brief reasoning for why this is a good choice"
    )
    return {"decision": response.content}


# -------------------------
# Build Graph (direct parallel edges)
# -------------------------

builder = StateGraph(State)

builder.add_node("practical_agent", practical_agent)
builder.add_node("creative_agent", creative_agent)
builder.add_node("cautious_agent", cautious_agent)
builder.add_node("tally_votes", tally_votes)

# Fan-out: all voters run in parallel
builder.add_edge(START, "practical_agent")
builder.add_edge(START, "creative_agent")
builder.add_edge(START, "cautious_agent")

# Fan-in: all votes flow to tally
builder.add_edge("practical_agent", "tally_votes")
builder.add_edge("creative_agent", "tally_votes")
builder.add_edge("cautious_agent", "tally_votes")
builder.add_edge("tally_votes", END)

graph = builder.compile()


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    print("ConsensusAI - Voting/Decision Agent\n")

    question = input("> What decision do you need help with? ")
    print("\nEnter your options (one per line, empty line to finish):")

    options = []
    while True:
        opt = input("  Option: ")
        if opt == "":
            break
        options.append(opt)

    if len(options) < 2:
        print("Need at least 2 options!")
    else:
        print("\n⏳ Agents are voting...\n")

        result = graph.invoke({"question": question, "options": options})

        print("=" * 60)
        print("VOTES & DECISION")
        print("=" * 60)
        print(result["decision"])
