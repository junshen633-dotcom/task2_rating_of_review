"""Parquet worker ownership and deterministic smoke selection checks."""
import sys
from pathlib import Path

import pytest
import torch

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import ParquetReviews


def fixture(directory):
    rows = [{"id": i, "rating": float(1 + i % 5), "title": "Five Stars", "comment": "Good food",
             "title_star": 5, "dense": [float(i)]} for i in range(23)]
    pq.write_table(pa.Table.from_pylist(rows), directory / "part-0.parquet", row_group_size=3)


def test_streaming_workers_never_duplicate(tmp_path):
    fixture(tmp_path)
    dataset = ParquetReviews(tmp_path, dense_dim=1, shuffle=True, seed=42, buffer_size=4)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)
    ids = [record["id"] for record in loader]
    assert sorted(ids) == list(range(23))
    assert len(set(ids)) == 23


def test_smoke_subset_independent_of_worker_count_and_epoch(tmp_path):
    fixture(tmp_path)
    observed = []
    for workers, epoch in ((0, 1), (2, 2)):
        dataset = ParquetReviews(tmp_path, dense_dim=1, shuffle=True, seed=42, epoch=epoch,
                                 max_rows=11, buffer_size=4)
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers)
        observed.append(sorted(record["id"] for record in loader))
    assert observed == [list(range(11)), list(range(11))]
