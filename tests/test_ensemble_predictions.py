import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ensemble_predictions import (corrected_losses, equal_weight, run, write_predictions)
from postprocess import Predictions, corrected_latent_risk


def prediction(ids, observed, latent, labels=None, titles=None):
    return Predictions(np.array(ids), np.array(observed, dtype=float),
                       np.array(titles or [0]*len(ids)),
                       np.array(labels, dtype=float) if labels is not None else None,
                       np.array(latent, dtype=float) if latent is not None else None)


def test_align_equal_weights_by_id_not_position():
    a = prediction([1, 2], [2, 4], [1, 5], [1, 5], [0, 5])
    b = prediction([2, 1], [3, 3], [4, 2], [5, 1], [5, 0])
    result = equal_weight([a, b])
    assert result.ids.tolist() == [1, 2]
    assert result.observed.tolist() == [2.5, 3.5]
    assert result.latent.tolist() == [1.5, 4.5]
    assert result.labels.tolist() == [1, 5]


@pytest.mark.parametrize("other, message", [
    (prediction([1, 3], [2, 4], [1, 5], [1, 5]), "ID sets"),
    (prediction([1, 2], [2, 4], [1, 5], [1, 4]), "labels differ"),
    (prediction([1, 2], [2, 4], [1, 5], [1, 5], [0, 5]), "title-star"),
    (prediction([1, 2], [2, 4], None, [1, 5]), "p_latent"),
])
def test_misaligned_members_rejected(other, message):
    with pytest.raises(ValueError, match=message):
        equal_weight([prediction([1, 2], [2, 4], [1, 5], [1, 5]), other])


def test_polynomial_corrected_losses_match_reference_non_symmetric():
    transition = np.array([[.8,.2,0,0,0], [0,.7,.3,0,0], [0,0,.9,.1,0],
                           [0,0,0,.6,.4], [.1,0,0,0,.9]])
    p, y = np.array([1.3, 3.9, 4.1]), np.array([1, 3, 5])
    result = corrected_losses(p, y, transition)
    reference = corrected_latent_risk(p, y, transition)
    assert result.mean() == pytest.approx(reference["estimated_latent_mse"])
    assert result.std(ddof=1)/np.sqrt(3) == pytest.approx(reference["standard_error"])


def setup_run(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"transition": np.eye(5).tolist()}), encoding="utf-8")
    sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
    for name, reverse in [("E4_seed43", False), ("E4", True)]:
        folder = tmp_path / "runs" / name
        folder.mkdir(parents=True)
        (folder / "metrics.json").write_text(json.dumps({"metadata_sha256": sha,
            "smoke": False, "evaluation_split_used_for_selection": False,
            "best_epoch": 3, "seed": 42 if reverse else 43}), encoding="utf-8")
        cal = (prediction([2, 1], [3, 3], [4, 2], [5, 1], [5, 0]) if reverse else
               prediction([1, 2], [2, 4], [1, 5], [1, 5], [0, 5]))
        test = (prediction([12, 11], [3, 3], [4, 2]) if reverse else
                prediction([11, 12], [2, 4], [1, 5]))
        write_predictions(folder / "predictions_cal.csv", cal)
        write_predictions(folder / "predictions_test.csv", test)
    sample = tmp_path / "sample.csv"
    sample.write_text("id,rating\n12,3\n11,3\n", encoding="utf-8")
    return argparse.Namespace(runs_root=tmp_path / "runs", models=["E4", "E4_seed43"],
                              name="E4_equal", baseline="E4_seed43", metadata=metadata,
                              sample=sample, out=tmp_path / "out")


def test_cli_pipeline_sample_order_and_overwrite_guard(tmp_path):
    args = setup_run(tmp_path)
    report = run(args)
    assert report["weights"] == [.5, .5]
    assert report["ensemble"]["conditional_corrected_latent_risk"]["estimated_latent_mse"] == pytest.approx(.25)
    with (args.out / "latent_hypothesis.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["id"]) for r in rows] == [12, 11]
    assert [float(r["rating"]) for r in rows] == [4.5, 1.5]
    with pytest.raises(FileExistsError):
        run(args)


def test_metadata_digest_mismatch_rejected_before_output(tmp_path):
    args = setup_run(tmp_path)
    path = args.runs_root / "E4" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["metadata_sha256"] = "wrong"
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="digest mismatch"):
        run(args)
    assert not args.out.exists()
