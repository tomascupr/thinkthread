import typer

app = typer.Typer()

@app.callback()
def callback():
    """
    CORT SDK - Command Line Interface
    """

@app.command()
def hello():
    """
    Simple test command to verify CLI functionality
    """
    print("Hello from CORT SDK CLI!")

if __name__ == "__main__":
    app()
