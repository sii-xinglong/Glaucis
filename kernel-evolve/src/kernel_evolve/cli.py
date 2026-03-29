"""CLI entry points for kernel-evolve."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from kernel_evolve import __version__
from kernel_evolve.config import load_config


@click.group()
@click.version_option(version=__version__)
def main():
  """kernel-evolve: Evolutionary TPU Kernel Optimizer."""


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--resume", type=click.Path(exists=True), help="Resume from a previous run directory.")
@click.option("--dry-run", is_flag=True, help="Validate config and exit without running.")
def run(config, resume, dry_run):
  """Start an evolution run."""
  cfg = load_config(config)
  click.echo(
    f"Loaded config: {cfg.kernel.name} ({len(cfg.shapes)} shapes, {cfg.evolution.max_generations} generations)"
  )

  if dry_run:
    click.echo("Dry run complete. Config is valid.")
    return

  config_dir = Path(config).resolve().parent
  template_path = config_dir / cfg.kernel.template
  reference_path = config_dir / cfg.kernel.reference

  if not template_path.exists():
    click.echo(f"Error: template file not found: {template_path}", err=True)
    sys.exit(1)
  if not reference_path.exists():
    click.echo(f"Error: reference file not found: {reference_path}", err=True)
    sys.exit(1)

  template_code = template_path.read_text()
  reference_code = reference_path.read_text()

  from kernel_evolve.llm import create_provider

  provider = create_provider(cfg.llm.provider.value, cfg.llm.model, cfg.llm.temperature)

  if cfg.evaluator.type.value == "ci":
    from kernel_evolve.ci_dispatcher import CIConfig, CIDispatcher

    ci_config = CIConfig(repo=cfg.evaluator.repo, workflow="kernel-eval.yaml")
    evaluator = CIDispatcher(ci_config)
  else:
    from kernel_evolve.kube_evaluator import KubeConfig, KubeEvaluator

    # Resolve job_template relative to repo root if not absolute
    job_template_path = Path(cfg.evaluator.job_template)
    if not job_template_path.is_absolute():
      repo_root = config_dir
      while repo_root != repo_root.parent:
        if (repo_root / ".git").exists():
          break
        repo_root = repo_root.parent
      job_template_path = repo_root / cfg.evaluator.job_template

    kube_config = KubeConfig(
      namespace=cfg.evaluator.namespace,
      job_template=str(job_template_path),
      repo=cfg.evaluator.repo,
      branch=cfg.evaluator.branch,
      poll_interval=cfg.evaluator.poll_interval,
      timeout=cfg.evaluator.timeout,
    )
    evaluator = KubeEvaluator(kube_config)

  from kernel_evolve.engine import EvolutionEngine

  engine = EvolutionEngine(
    config=cfg,
    provider=provider,
    evaluator=evaluator,
    template_code=template_code,
    reference_code=reference_code,
  )

  click.echo(f"Starting evolution: {cfg.evolution.num_islands} islands, {cfg.evolution.population_size} population")
  best = asyncio.run(engine.run())

  if best:
    click.echo(f"Best kernel: {best.id} (fitness: {best.fitness:.2f}x)")
    click.echo(f"Saved to: {cfg.logging.output_dir}/best/kernel.py")
  else:
    click.echo("No successful variants found.")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
def status(run_dir):
  """Show progress of an evolution run."""
  run_path = Path(run_dir)
  archive_files = list(run_path.glob("population/archive_island_*.json"))

  if not archive_files:
    click.echo(f"No archive found in {run_dir}")
    return

  from kernel_evolve.population import Archive

  total_variants = 0
  best_fitness = 0.0
  best_id = ""

  for af in archive_files:
    archive = Archive.load(af)
    total_variants += archive.size
    if archive.best and archive.best.fitness > best_fitness:
      best_fitness = archive.best.fitness
      best_id = archive.best.id

  click.echo(f"Islands: {len(archive_files)} | Variants: {total_variants} | Best: {best_id} ({best_fitness:.2f}x)")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
def best(run_dir):
  """Extract the best kernel from a completed run."""
  best_path = Path(run_dir) / "best" / "kernel.py"
  if not best_path.exists():
    click.echo("No best kernel found.")
    return
  click.echo(best_path.read_text())
