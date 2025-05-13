"""Test script for TreeThinker CLI.

This script demonstrates how to use the TreeThinker CLI to solve problems
using tree-of-thoughts reasoning and visualize the thinking tree.
"""

import os
import subprocess
import sys
from pathlib import Path

os.environ["OPENAI_API_KEY"] = (
    "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"
)

repo_root = Path(__file__).parent.parent.absolute()

sys.path.insert(0, str(repo_root))

from thinkthread_sdk.cli import app


def run_cli_command(command):
    """Run a CLI command and print the output."""
    print(f"\n=== Running command: {command} ===\n")

    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    print(result.stdout)

    if result.returncode != 0:
        print("\nCommand failed with the following errors:")
        print(result.stderr)

    print("\n=== Command completed ===\n")


def main():
    """Run test commands for the TreeThinker CLI."""
    print("Testing TreeThinker CLI...")

    run_cli_command(
        'python -m thinkthread_sdk tot "What are three key benefits of tree-based search for reasoning?" '
        "--provider openai --beam-width 2 --max-depth 2 --iterations 1"
    )

    run_cli_command(
        'python -m thinkthread_sdk tot "How can we solve the climate crisis?" '
        "--provider openai --beam-width 2 --max-depth 2 --iterations 1"
    )

    run_cli_command(
        'python -m thinkthread_sdk tot "What is the meaning of life?" '
        "--provider dummy --beam-width 2 --max-depth 2 --iterations 1"
    )

    print("TreeThinker CLI test completed successfully!")


if __name__ == "__main__":
    main()
