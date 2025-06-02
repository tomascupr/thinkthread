"""Simple CLI for ThinkThread."""

import typer
from rich.console import Console

from . import reason
from . import explore as explore_fn
from . import solve as solve_fn  
from . import debate as debate_fn
from . import refine as refine_fn
from . import __version__

app = typer.Typer(
    name="think",
    help="ThinkThread - Make your AI think before it speaks",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
):
    """ThinkThread - Make your AI think before it speaks."""
    if version:
        console.print(f"ThinkThread v{__version__}")
        raise typer.Exit(0)

    # If no command and no args, show help
    if ctx.invoked_subcommand is None and not ctx.args:
        console.print("[yellow]Usage: think <question>[/yellow]")
        console.print("\nExamples:")
        console.print('  think "What are the implications of quantum computing?"')
        console.print('  think explore "Creative solutions for climate change"')
        console.print('  think solve "Our deployment takes 45 minutes"')
        raise typer.Exit(1)


@app.command(context_settings={"allow_extra_args": True})
def default(
    ctx: typer.Context,
    test: bool = typer.Option(False, "--test", "-t", help="Test mode - no API calls"),
):
    """Process a question with automatic mode selection."""
    if not ctx.args:
        console.print("[red]Error: Please provide a question[/red]")
        raise typer.Exit(1)

    # Filter out 'default' from args if it's there
    args = [arg for arg in ctx.args if arg != "default"]
    question = " ".join(args)

    with console.status("[bold green]Thinking...", spinner="dots"):
        answer = reason(question, test_mode=test)
    console.print(answer)


@app.command()
def explore(
    question: str = typer.Argument(..., help="Question to explore"),
    test: bool = typer.Option(False, "--test", "-t", help="Test mode - no API calls"),
):
    """Explore ideas using Tree-of-Thoughts."""
    with console.status("[bold green]Exploring ideas...", spinner="dots"):
        answer = explore_fn(question, test_mode=test)
    console.print(answer)


@app.command()
def solve(
    problem: str = typer.Argument(..., help="Problem to solve"),
    test: bool = typer.Option(False, "--test", "-t", help="Test mode - no API calls"),
):
    """Get step-by-step solutions."""
    with console.status("[bold green]Finding solutions...", spinner="dots"):
        answer = solve_fn(problem, test_mode=test)
    console.print(answer)


@app.command()
def debate(
    question: str = typer.Argument(..., help="Question to analyze"),
    test: bool = typer.Option(False, "--test", "-t", help="Test mode - no API calls"),
):
    """Analyze from multiple perspectives."""
    with console.status("[bold green]Analyzing perspectives...", spinner="dots"):
        answer = debate_fn(question, test_mode=test)
    console.print(answer)


@app.command()
def refine(
    text: str = typer.Argument(..., help="Text to refine"),
    instructions: str = typer.Option(
        "", "--instructions", "-i", help="Specific refinement instructions"
    ),
    test: bool = typer.Option(False, "--test", "-t", help="Test mode - no API calls"),
):
    """Refine and improve text."""
    with console.status("[bold green]Refining...", spinner="dots"):
        answer = refine_fn(text, instructions, test_mode=test)
    console.print(answer)


if __name__ == "__main__":
    app()
