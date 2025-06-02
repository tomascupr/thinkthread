"""ThinkThread CLI - Simple, powerful reasoning at your fingertips."""

import typer
from typing import Optional
from rich import print
from rich.console import Console
from rich.panel import Panel

from thinkthread import reason, explore, refine, debate, solve
from thinkthread import __version__

app = typer.Typer(
    name="think",
    help="ThinkThread - Make your AI think before it speaks"
)
console = Console()


def think_logic(
    question: str,
    mode: Optional[str] = None,
    visualize: bool = False,
    debug: bool = False,
    test: bool = False,
    profile: bool = False,
    stream: bool = False,
    confidence: float = 0.7,
    max_cost: float = 0.50,
):
    """Core thinking logic used by both default and think command."""
    # Build kwargs for reasoning functions
    kwargs = {
        "test_mode": test,
        "stream": stream,
        "confidence_threshold": confidence,
        "max_cost": max_cost
    }
    
    # Enable visualization/debugging if requested
    if visualize:
        kwargs["visualize"] = True
    if debug:
        kwargs["debug"] = True
    if profile:
        kwargs["profile"] = True
    
    try:
        # Select reasoning function
        if mode:
            modes = {
                "explore": explore,
                "refine": refine,
                "debate": debate,
                "solve": solve
            }
            if mode not in modes:
                console.print(f"[red]Unknown mode: {mode}[/red]")
                console.print("Available modes: explore, refine, debate, solve")
                raise typer.Exit(1)
            reasoning_fn = modes[mode]
        else:
            reasoning_fn = reason
        
        # Show thinking indicator
        if not stream:
            with console.status("[bold green]Thinking...", spinner="dots"):
                answer = reasoning_fn(question, **kwargs)
        else:
            answer = reasoning_fn(question, **kwargs)
        
        # Display answer
        if hasattr(answer, '__str__'):
            console.print(Panel(str(answer), title="[bold]Answer[/bold]", expand=False))
        else:
            console.print(answer)
        
        # Show metadata if available
        if hasattr(answer, 'cost') and answer.cost:
            console.print(f"\n💰 Cost: [green]${answer.cost:.4f}[/green]")
        if hasattr(answer, 'confidence') and answer.confidence:
            console.print(f"🎯 Confidence: [blue]{answer.confidence:.0%}[/blue]")
        if hasattr(answer, 'reasoning_mode'):
            console.print(f"🧠 Mode: [yellow]{answer.reasoning_mode}[/yellow]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Thinking interrupted[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    question: Optional[str] = typer.Argument(None, help="The question to think about"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Reasoning mode: explore, refine, debate, solve"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Open live reasoning visualization"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Show detailed reasoning steps"),
    test: bool = typer.Option(False, "--test", "-t", help="Test mode - no API calls"),
    profile: bool = typer.Option(False, "--profile", "-p", help="Show performance profiling"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream output as it's generated"),
    confidence: float = typer.Option(0.7, "--confidence", "-c", help="Minimum confidence threshold"),
    max_cost: float = typer.Option(0.50, "--max-cost", help="Maximum cost per query"),
    version: bool = typer.Option(False, "--version", "-V", help="Show version and exit"),
):
    """ThinkThread - Make your AI think before it speaks."""
    if version:
        console.print(f"ThinkThread v{__version__}")
        raise typer.Exit(0)
    
    # If no subcommand and we have a question, run think
    if ctx.invoked_subcommand is None:
        if question:
            think_logic(question, mode, visualize, debug, test, profile, stream, confidence, max_cost)
        else:
            # Show help if no question provided
            console.print("[yellow]No question provided. Use --help for usage information.[/yellow]")
            raise typer.Exit(1)


@app.command()
def think(
    question: str = typer.Argument(..., help="The question to think about"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Reasoning mode: explore, refine, debate, solve"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Open live reasoning visualization"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Show detailed reasoning steps"),
    test: bool = typer.Option(False, "--test", "-t", help="Test mode - no API calls"),
    profile: bool = typer.Option(False, "--profile", "-p", help="Show performance profiling"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream output as it's generated"),
    confidence: float = typer.Option(0.7, "--confidence", "-c", help="Minimum confidence threshold"),
    max_cost: float = typer.Option(0.50, "--max-cost", help="Maximum cost per query"),
):
    """Think about any question using advanced reasoning."""
    think_logic(question, mode, visualize, debug, test, profile, stream, confidence, max_cost)


@app.command()
def modes():
    """List available reasoning modes."""
    mode_info = {
        "explore": "🌳 Tree of Thoughts - Best for creative tasks and brainstorming",
        "refine": "🔄 Recursive refinement - Best for improving and polishing content",
        "debate": "🎭 Multi-perspective - Best for balanced analysis and decisions",
        "solve": "🎯 Solution-focused - Best for specific problems and action plans",
        "auto": "🤖 Automatic selection - Let ThinkThread choose the best mode"
    }
    
    console.print(Panel.fit("[bold]Available Reasoning Modes[/bold]"))
    for mode, description in mode_info.items():
        console.print(f"\n[bold]{mode}[/bold]: {description}")


@app.command()
def quick(
    question: str = typer.Argument(..., help="Question for quick exploration"),
):
    """Quick brainstorming with explore mode."""
    with console.status("[bold green]Exploring ideas...", spinner="dots"):
        answer = explore(question, test_mode=False, stream=False)
    console.print(Panel(str(answer), title="[bold]Ideas[/bold]", expand=False))


@app.command()
def compare(
    question: str = typer.Argument(..., help="Question to analyze from multiple angles"),
):
    """Compare different perspectives with debate mode."""
    with console.status("[bold green]Analyzing perspectives...", spinner="dots"):
        answer = debate(question, test_mode=False, stream=False)
    console.print(Panel(str(answer), title="[bold]Analysis[/bold]", expand=False))


@app.command()
def fix(
    problem: str = typer.Argument(..., help="Problem to solve"),
):
    """Get actionable solutions with solve mode."""
    with console.status("[bold green]Finding solutions...", spinner="dots"):
        answer = solve(problem, test_mode=False, stream=False)
    console.print(Panel(str(answer), title="[bold]Solution[/bold]", expand=False))


@app.command()
def polish(
    text: str = typer.Argument(..., help="Text to refine and improve"),
):
    """Polish and improve text with refine mode."""
    with console.status("[bold green]Refining...", spinner="dots"):
        answer = refine(text, test_mode=False, stream=False)
    console.print(Panel(str(answer), title="[bold]Refined Text[/bold]", expand=False))


if __name__ == "__main__":
    app()