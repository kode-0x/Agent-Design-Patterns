"""
Pattern: Single Agent
Project: AskBot - A simple chatbot using OpenAI Agents SDK

This is the simplest agent pattern - one agent handles everything.
Uses run_demo_loop for a built-in interactive REPL with conversation history.
"""

import asyncio
from agents import Agent, run_demo_loop

agent = Agent(
    name="AskBot",
    instructions="""You are a helpful, friendly, and knowledgeable AI assistant.

Guidelines:
- Answer questions accurately and clearly.
- Be concise but provide detail when needed.
- Explain technical concepts in simple language.
- If unsure about something, say so honestly.
""",
)


async def main():
    await run_demo_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())
