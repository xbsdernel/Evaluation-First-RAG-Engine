"""
CLI entry point for the RAG Engine evaluation harness.

Usage
-----
    rag --help
    rag evaluate --help   # populated in future issues

The CLI is structured as a Click group so each pipeline stage can register
its own sub-command without touching this file.
"""

from __future__ import annotations

import random
import sys

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(package_name="rag-engine")
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Global RNG seed for reproducible runs.",
)
@click.pass_context
def main(ctx: click.Context, seed: int) -> None:
    """Evaluation-First RAG Engine — harness CLI.

    Run ``rag COMMAND --help`` for sub-command details.
    """
    ctx.ensure_object(dict)
    ctx.obj["seed"] = seed
    _seed_global_rng(seed)


def _seed_global_rng(seed: int) -> None:
    """Seed the Python stdlib RNG. NumPy / PyTorch seeded in their own modules."""
    random.seed(seed)


# ---------------------------------------------------------------------------
# Placeholder sub-commands (expanded in later issues)
# ---------------------------------------------------------------------------

@main.command()
@click.pass_context
def evaluate(ctx: click.Context) -> None:
    """Run the evaluation harness against a retrieval run. (not yet implemented)"""
    console.print("[yellow]'evaluate' sub-command will be implemented in ISS-013.[/yellow]")
    sys.exit(0)


@main.command()
@click.pass_context
def reproduce(ctx: click.Context) -> None:
    """Reproduce headline results from a run ID. (not yet implemented)"""
    console.print("[yellow]'reproduce' sub-command will be implemented in ISS-045.[/yellow]")
    sys.exit(0)


if __name__ == "__main__":
    main()
