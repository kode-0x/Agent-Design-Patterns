# Agent Design Patterns

A collection of 20 AI agent design patterns, each implemented as a simple, self-contained Python project for learning.

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` and add your API keys:
```
OPENAI_API_KEY=your_key
OPENROUTER_API_KEY=your_key
TAVILY_API_KEY=your_key
```

## Patterns

| # | Pattern | Project | Technologies |
|---|---------|---------|--------------|
| 1 | Single Agent | AskBot - Chatbot | OpenAI Agents SDK |
| 2 | Sequential Agent | BlogFlow - Blog Generator | LangGraph + OpenRouter |
| 3 | Parallel Agents | ResearchHub - Research Agent | LangGraph + Tavily + OpenRouter |
| 4 | Sequential + Parallel | NewsFlow - News Summarizer | LangGraph + Tavily + OpenRouter |
| 5 | Orchestrator–Worker | TaskMaster AI - Coding Task Manager | LangGraph + OpenRouter |
| 6 | Router / Classifier | MailRouter - Email Classifier | LangGraph + OpenRouter |
| 7 | Handoff | SupportFlow - Customer Support Bot | LangGraph + OpenRouter |
| 8 | Agent as Tool | ToolBot - Calculator Assistant | LangGraph + Python Tools |
| 9 | Planner–Executor | PlanDo AI - To-Do List Planner | LangGraph + SQLite |
| 10 | Review & Critique | CritiqueAI - Essay Checker | LangGraph + OpenRouter |
| 11 | Review & Critique Loop | ImproveLoop - Essay Improver | LangGraph + OpenRouter |
| 12 | Reflection / Self-Critique | SelfCheck AI - Code Improver | LangGraph + OpenRouter |
| 13 | Debate / Multi-Agent | DebateBot - Pros & Cons Generator | LangGraph + OpenRouter |
| 14 | Consensus | ConsensusAI - Voting/Decision Agent | LangGraph + OpenRouter |
| 15 | Hierarchical Agent | TaskTree AI - Simple Task Manager | LangGraph + SQLite |
| 16 | Human-in-the-Loop | HumanApprove - Email Approval Bot | LangGraph + OpenRouter |
| 17 | Event-Driven Agent | EventBot - Reminder Bot | LangGraph + OpenRouter |
| 18 | Shared State / Blackboard | SharedBrain - Shared Study Notes | LangGraph + SQLite |
| 19 | Planner–Executor–Reviewer | PlanBuildReview - Task Completion System | LangGraph + SQLite |

## How to Run

Each pattern is in its own folder. Run any agent with:

```bash
python "Single-Agent/Chatbot.py"
python "Sequential-Agent/Blog-Generator.py"
python "Parallel Agents/Research-Agent.py"
# ... etc
```
