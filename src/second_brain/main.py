"""Application entry point: CLI chat interface."""

from __future__ import annotations

import click
from rich.console import Console
from rich.markdown import Markdown

from second_brain.config.settings import get_settings
from second_brain.core.agent import Agent
from second_brain.ltm.db_manager import DatabaseManager
from second_brain.ltm.embedder import Embedder
from second_brain.ltm.retriever import Retriever

console = Console()


def build_agent() -> tuple[Agent, DatabaseManager]:
    """Wire up all components and return the agent + db handle for cleanup."""
    settings = get_settings()
    db = DatabaseManager(settings.db)
    db.connect()
    embedder = Embedder(settings.embedding)
    retriever = Retriever(db=db, embedder=embedder, settings=settings)
    agent = Agent(retriever=retriever, settings=settings)
    return agent, db


@click.group()
def cli() -> None:
    """Second Brain - Your personal knowledge assistant."""


@cli.command()
def chat() -> None:
    """Start an interactive chat session with your Second Brain."""
    agent, db = build_agent()
    console.print("[bold green]Second Brain[/bold green] ready. Type 'quit' to exit.\n")

    try:
        while True:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            with console.status("Thinking..."):
                response = agent.chat(user_input)
            console.print("\n[bold yellow]Brain:[/bold yellow]")
            console.print(Markdown(response))
            console.print()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        db.close()
        console.print("\n[dim]Session ended.[/dim]")


@cli.command()
def stats() -> None:
    """Show memory statistics."""
    settings = get_settings()
    with DatabaseManager(settings.db) as db:
        count = db.count()
    console.print(f"Total memories in LTM: [bold]{count}[/bold]")


if __name__ == "__main__":
    cli()
