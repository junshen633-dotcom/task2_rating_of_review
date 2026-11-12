import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from postprocess import (align_to_sample, apply_shrinkage, affine_transition,
                         corrected_latent_risk, fit_shrinkage, read_predictions,
                         run, validate_transition)


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def test_shrink_known_scale_and_constant():
    p = np.array([1., 2., 3., 4., 5.])
    y = 3 + .5 * (p - 3)
    fit = fit_shrinkage(p, y)
    assert fit["anchor"] == 3
    assert fit["s"] == pytest.approx(.5)
    assert np.allclose(apply_shrinkage(p, fit), y)
    assert fit_shrinkage(np.full(4, 3.), np.array([2., 3., 3., 4.]))["s"] == 0


def test_identity_risk_is_actual_mse():
    p = np.array([1.3, 3.2, 4.7])
    y = np.array([1, 4, 5])
    risk = corrected_latent_risk(p, y, np.eye(5))
    assert risk["available"]
    assert risk["estimated_latent_mse"] == pytest.approx(np.mean((p-y)**2))


def test_transition_direction_non_symmetric():
    # P(observed | latent), deliberately nonsymmetric to catch a transpose bug.
    transition = np.array([[.8,.2,0,0,0], [0,.7,.3,0,0], [0,0,.9,.1,0],
                           [0,0,0,.6,.4], [.1,0,0,0,.9]])
    p = 2.4
    corrected = np.array([corrected_latent_risk(np.array([p]), np.array([j]), transition)
                          ["estimated_latent_mse"] for j in range(1, 6)])
    assert np.allclose(transition @ corrected, (np.arange(1, 6)-p)**2)


def test_negative_risk_is_not_clamped_or_square_rooted():
    transition = .75*np.eye(5) + .05*np.ones((5,5))
    result = corrected_latent_risk(np.array([3.]), np.array([3]), transition)
    assert result["estimated_latent_mse"] < 0
    assert "rmse" not in result


def test_ill_conditioned_risk_refused():
    result = corrected_latent_risk(np.array([3.]), np.array([3]), np.ones((5,5))/.5/10)
    assert not result["available"]


def test_affine_map_matches_symmetric_noise():
    transition = .75*np.eye(5) + .05*np.ones((5,5))
    result = affine_transition(transition)
    assert result["allowed"]
    assert result["slope"] == pytest.approx(.75)
    assert result["intercept"] == pytest.approx(.75)
    assert not affine_transition(np.ones((5,5))/5)["allowed"]


def test_wrong_ids_reordering_and_finite_guard():
    assert np.allclose(align_to_sample([2,0,1],[4,2,3],[0,1,2]), [2,3,4])
    with pytest.raises(ValueError, match="id sets differ"):
        align_to_sample([1,2],[2,3],[0,1])
    with pytest.raises(ValueError, match="Duplicate"):
        align_to_sample([1,1],[2,3],[1,2])
    with pytest.raises(ValueError, match="finite"):
        align_to_sample([0],[np.nan],[0])


def test_prediction_reader_rejects_bad_inputs(tmp_path):
    fields = ["id","rating","title_star","p_observed"]
    path = tmp_path / "pred.csv"
    write_csv(path, fields, [[0,5,5,"nan"]])
    with pytest.raises(ValueError, match="finite"):
        read_predictions(path, labelled=True)
    write_csv(path, fields, [[0,5,5,5.1]])
    with pytest.raises(ValueError, match="outside"):
        read_predictions(path, labelled=True)
    write_csv(path, fields, [[0,5,5,4],[0,1,1,2]])
    with pytest.raises(ValueError, match="Duplicate"):
        read_predictions(path, labelled=True)
    with pytest.raises(ValueError, match="sum to one"):
        validate_transition(np.eye(5)*.9)


def test_end_to_end_eval_does_not_change_calibration(tmp_path):
    fields = ["id","rating","title_star","p_observed"]
    cal = tmp_path/"cal.csv"
    evaluation = tmp_path/"eval.csv"
    test = tmp_path/"test.csv"
    sample = tmp_path/"sample.csv"
    metadata = tmp_path/"metadata.json"
    write_csv(cal, fields, [[10,2,2,1],[11,3,3,3],[12,4,4,5]])
    write_csv(evaluation, fields, [[20,1,1,2],[21,5,5,4]])
    write_csv(test, fields, [[2,"",0,5],[0,"",0,1],[1,"",0,3]])
    write_csv(sample, ["id","rating"], [[0,3],[1,3],[2,3]])
    metadata.write_text(json.dumps({"transition":np.eye(5).tolist()}))
    args = argparse.Namespace(cal=str(cal), eval=str(evaluation), test=str(test),
                              sample=str(sample), metadata=str(metadata), out=str(tmp_path/"out"))
    calibration, report = run(args)
    with open(tmp_path/"out"/"observed_calibrated.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["id"] for r in rows] == ["0","1","2"]
    assert [float(r["rating"]) for r in rows] == [2.,3.,4.]
    assert report["evaluation"] is not None
    write_csv(evaluation, fields, [[20,5,1,2],[21,1,5,4]])
    args.out = str(tmp_path/"out2")
    calibration2, _ = run(args)
    assert calibration == calibration2


def test_cal_eval_overlap_rejected(tmp_path):
    fields = ["id","rating","title_star","p_observed"]
    for name in ["cal","eval","test"]:
        write_csv(tmp_path/f"{name}.csv", fields, [[0,3,3,3]])
    args = argparse.Namespace(cal=tmp_path/"cal.csv", eval=tmp_path/"eval.csv",
                              test=tmp_path/"test.csv", sample="unused", metadata="unused",
                              out=tmp_path/"out")
    with pytest.raises(ValueError, match="overlap"):
        run(args)
