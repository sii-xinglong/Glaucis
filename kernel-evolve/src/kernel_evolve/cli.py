"""CLI entry points for kernel-evolve."""

import click


@click.group()
@click.version_option()
def main():
  """Evolutionary TPU Kernel Optimizer."""


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--resume", type=click.Path(exists=True), help="Resume from a previous run directory.")
def run(config, resume):
  """Start an evolution run."""
  click.echo(f"Starting evolution with config: {config}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
def status(run_dir):
  """Show progress of an evolution run."""
  click.echo(f"Status for: {run_dir}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
def best(run_dir):
  """Extract the best kernel from a completed run."""
  click.echo(f"Best kernel from: {run_dir}")
