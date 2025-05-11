import typer
from cort_sdk import __version__

app = typer.Typer()

@app.callback()
def callback():
    """
    CORT SDK - Command Line Interface
    """

@app.command()
def version():
    """
    Show the current version of CORT SDK
    """
    print(f"CORT SDK version: {__version__}")

if __name__ == "__main__":
    app()
