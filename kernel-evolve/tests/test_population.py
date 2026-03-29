"""Tests for MAP-Elites population manager."""

import pytest

from kernel_evolve.population import Archive, BehaviorDescriptor, Variant


@pytest.fixture
def archive():
  return Archive(
    descriptor_axes={
      "block_size": [64, 128, 256, 512],
      "pipeline_stages": [1, 2, 3, 4],
      "memory_strategy": ["scratch", "hbm", "rmw"],
      "compute_profile": ["very_low", "low", "medium", "high"],
    }
  )


def test_archive_empty_on_creation(archive):
  assert archive.size == 0
  assert archive.best is None
  assert archive.capacity == 4 * 4 * 3 * 4  # 192 cells


def test_insert_variant(archive):
  desc = BehaviorDescriptor(block_size=128, pipeline_stages=2, memory_strategy="scratch")
  variant = Variant(
    id="v001",
    code="def kernel(): pass",
    descriptor=desc,
    fitness=1.5,
    generation=1,
    parent_id=None,
  )
  inserted = archive.insert(variant)
  assert inserted is True
  assert archive.size == 1
  assert archive.best.id == "v001"


def test_insert_fitter_replaces(archive):
  desc = BehaviorDescriptor(block_size=128, pipeline_stages=2, memory_strategy="scratch")
  v1 = Variant(id="v001", code="v1", descriptor=desc, fitness=1.5, generation=1, parent_id=None)
  v2 = Variant(id="v002", code="v2", descriptor=desc, fitness=2.0, generation=2, parent_id="v001")
  archive.insert(v1)
  archive.insert(v2)
  assert archive.size == 1  # same cell, replaced
  assert archive.best.id == "v002"


def test_insert_weaker_rejected(archive):
  desc = BehaviorDescriptor(block_size=128, pipeline_stages=2, memory_strategy="scratch")
  v1 = Variant(id="v001", code="v1", descriptor=desc, fitness=2.0, generation=1, parent_id=None)
  v2 = Variant(id="v002", code="v2", descriptor=desc, fitness=1.0, generation=2, parent_id="v001")
  archive.insert(v1)
  inserted = archive.insert(v2)
  assert inserted is False
  assert archive.best.id == "v001"


def test_tournament_select(archive):
  for i, bs in enumerate([64, 128, 256]):
    desc = BehaviorDescriptor(block_size=bs, pipeline_stages=1, memory_strategy="scratch")
    v = Variant(id=f"v{i}", code=f"c{i}", descriptor=desc, fitness=float(i + 1), generation=1, parent_id=None)
    archive.insert(v)
  selected = archive.tournament_select(k=2, rng_seed=42)
  assert selected.id in {"v0", "v1", "v2"}


def test_checkpoint_and_restore(archive, tmp_path):
  desc = BehaviorDescriptor(block_size=128, pipeline_stages=2, memory_strategy="scratch")
  v = Variant(id="v001", code="code", descriptor=desc, fitness=1.5, generation=1, parent_id=None)
  archive.insert(v)

  checkpoint_path = tmp_path / "archive.json"
  archive.save(checkpoint_path)

  restored = Archive.load(checkpoint_path)
  assert restored.size == 1
  assert restored.best.id == "v001"
  assert restored.best.fitness == 1.5


def test_all_variants(archive):
  for i, bs in enumerate([64, 128]):
    desc = BehaviorDescriptor(block_size=bs, pipeline_stages=1, memory_strategy="scratch")
    v = Variant(id=f"v{i}", code=f"c{i}", descriptor=desc, fitness=float(i + 1), generation=1, parent_id=None)
    archive.insert(v)
  variants = archive.all_variants()
  assert len(variants) == 2


def test_compute_profile_descriptor():
  desc = BehaviorDescriptor(block_size=128, pipeline_stages=2, memory_strategy="scratch", compute_profile="high")
  assert desc.cell_key() == (128, 2, "scratch", "high")


def test_compute_profile_default():
  desc = BehaviorDescriptor(block_size=128, pipeline_stages=2, memory_strategy="scratch")
  assert desc.compute_profile == "medium"
  assert desc.cell_key() == (128, 2, "scratch", "medium")


def test_archive_capacity_with_compute_profile():
  archive = Archive()
  assert archive.capacity == 4 * 4 * 3 * 4  # 192 cells


def test_ratio_to_profile_bucket():
  from kernel_evolve.population import ratio_to_compute_profile

  assert ratio_to_compute_profile(0.1) == "very_low"
  assert ratio_to_compute_profile(0.3) == "low"
  assert ratio_to_compute_profile(0.6) == "medium"
  assert ratio_to_compute_profile(0.9) == "high"
  assert ratio_to_compute_profile(None) == "medium"
  assert ratio_to_compute_profile(0.0) == "very_low"
  assert ratio_to_compute_profile(1.0) == "high"
  assert ratio_to_compute_profile(0.25) == "low"
  assert ratio_to_compute_profile(0.75) == "high"
