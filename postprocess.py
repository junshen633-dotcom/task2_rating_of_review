#!/usr/bin/env python3
"""Calibrate on cal only, audit on eval, and align candidate files by sample id.

Latent candidates and inverse-transition risk are explicitly conditional on the
title-proxy transition being the true label-noise mechanism. They are not known
clean labels, guaranteed better predictions, or estimates of leaderboard score.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


MAX_TRANSITION_CONDITION = 100.0
MAX_AFFINE_DEVIATION = 0.10
MIN_AFFINE_SLOPE = 0.20


@dataclass
class Predictions:
    ids: np.ndarray
    observed: np.ndarray
    title_star: np.ndarray
    labels: np.ndarray | None = None
    latent: np.ndarray | None = None


def numeric(value, name, low=None, high=None):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric: {value!r}") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if low is not None and number < low or high is not None and number > high:
        raise ValueError(f"{name} outside [{low}, {high}]: {number}")
    return number


def parse_id(value):
    if not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError(f"id must be a nonnegative integer: {value!r}")
    return int(value)


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"Missing or duplicate column names: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row length: {path}")
    return fields, rows


def unique_ids(rows, path):
    ids = [parse_id(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate id: {path}")
    return np.asarray(ids, dtype=np.int64)


def read_predictions(path, labelled=False):
    fields, rows = read_rows(path)
    required = {"id", "title_star", "p_observed"} | ({"rating"} if labelled else set())
    if not required.issubset(fields):
        raise ValueError(f"{path}: missing columns {sorted(required - set(fields))}")
    ids = unique_ids(rows, path)
    observed = np.asarray([numeric(r["p_observed"], "p_observed", 1, 5) for r in rows])
    title = np.asarray([numeric(r["title_star"] or 0, "title_star", 0, 5) for r in rows])
    if np.any(title != np.floor(title)):
        raise ValueError("title_star must be integer 0 (unknown) or 1..5")
    labels = None
    if labelled:
        labels = np.asarray([numeric(r["rating"], "rating", 1, 5) for r in rows])
        if np.any(labels != np.floor(labels)):
            raise ValueError("Observed ratings must be integers 1..5")
    latent = None
    if "p_latent" in fields:
        values = [r["p_latent"] for r in rows]
        if any(v != "" for v in values):
            latent = np.asarray([numeric(v, "p_latent", 1, 5) for v in values])
    return Predictions(ids, observed, title.astype(np.int64), labels, latent)


def read_sample(path):
    fields, rows = read_rows(path)
    if fields != ["id", "rating"]:
        raise ValueError("Sample must have exactly the ordered columns id,rating")
    ids = unique_ids(rows, path)
    for row in rows:
        numeric(row["rating"], "sample rating", 1, 5)
    return ids


def validate_transition(transition):
    transition = np.asarray(transition, dtype=np.float64)
    if transition.shape != (5, 5) or not np.isfinite(transition).all():
        raise ValueError("transition must be a finite 5x5 matrix")
    if np.any(transition < 0) or np.any(transition > 1):
        raise ValueError("transition entries must be probabilities")
    if not np.allclose(transition.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("transition rows (latent -> observed) must sum to one")
    return transition


def fit_shrinkage(predictions, labels):
    """Fit anchor and scalar shrink only from calibration examples."""
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if predictions.shape != labels.shape or predictions.ndim != 1 or len(labels) == 0:
        raise ValueError("Calibration vectors must be nonempty, one-dimensional and aligned")
    if not np.isfinite(predictions).all() or not np.isfinite(labels).all():
        raise ValueError("Nonfinite calibration data")
    anchor = float(labels.mean())
    b = predictions - anchor
    denominator = float(b @ b)
    coefficient = 0.0 if denominator < 1e-12 else float(
        np.clip((b @ (labels - anchor)) / denominator, 0, 1))
    return {"anchor": anchor, "s": coefficient,
            "anchor_source": "calibration_label_mean", "n_calibration": len(labels),
            "denominator": denominator}


def apply_shrinkage(predictions, calibration):
    return np.clip(calibration["anchor"] + calibration["s"] *
                   (np.asarray(predictions) - calibration["anchor"]), 1, 5)


def affine_transition(transition):
    """Approximate E[observed|latent=z] = slope*z + intercept using training T."""
    transition = validate_transition(transition)
    z = np.arange(1, 6, dtype=np.float64)
    means = transition @ z
    slope, intercept = np.linalg.lstsq(np.column_stack([z, np.ones(5)]), means,
                                     rcond=None)[0]
    deviation = float(np.max(np.abs(means - (slope * z + intercept))))
    condition = float(np.linalg.cond(transition))
    allowed = bool(slope > MIN_AFFINE_SLOPE and deviation <= MAX_AFFINE_DEVIATION
                   and condition <= MAX_TRANSITION_CONDITION)
    return {"slope": float(slope), "intercept": float(intercept),
            "row_expected_observed": means.tolist(), "max_absolute_deviation": deviation,
            "condition_number": condition if np.isfinite(condition) else None,
            "allowed": allowed,
            "criteria": {"slope_greater_than": MIN_AFFINE_SLOPE,
                         "max_absolute_deviation_at_most": MAX_AFFINE_DEVIATION,
                         "condition_number_at_most": MAX_TRANSITION_CONDITION}}


def latent_candidate(predictions, transform):
    if predictions.latent is not None:
        return predictions.latent.copy(), "model_p_latent"
    if not transform["allowed"]:
        return None, "skipped: training transition is not sufficiently affine/invertible"
    return (np.clip((predictions.observed - transform["intercept"]) /
                    transform["slope"], 1, 5), "inverse_affine_training_transition")


def metrics(predictions, targets):
    residual = np.asarray(predictions, dtype=np.float64) - np.asarray(targets)
    return {"n": len(residual), "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "prediction_mean": float(np.mean(predictions)),
            "prediction_std": float(np.std(predictions))}


def corrected_latent_risk(predictions, observed_labels, transition):
    """Return mean inverse-noise squared loss, not sqrt or a guaranteed clean risk.

    T[z,j] = P(observed=j | latent=z). Let v[z]=(z-p)^2. Solve T*r=v,
    and score r[observed]. E[r[observed] | z,p] = v[z] only when T is the
    correct class-conditional, feature-independent noise process.
    """
    transition = validate_transition(transition)
    condition = float(np.linalg.cond(transition))
    if not np.isfinite(condition) or condition > MAX_TRANSITION_CONDITION:
        return {"available": False, "reason": "transition is ill-conditioned or singular"}
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(observed_labels)
    if predictions.shape != labels.shape or predictions.ndim != 1 or not len(labels):
        raise ValueError("Risk vectors must be nonempty and aligned")
    if not np.isfinite(predictions).all() or not np.isfinite(labels).all():
        raise ValueError("Nonfinite risk input")
    if np.any(labels != np.floor(labels)) or np.any((labels < 1) | (labels > 5)):
        raise ValueError("Risk observed labels must be 1..5 integers")
    squared_loss = (np.arange(1, 6)[None, :] - predictions[:, None]) ** 2
    corrected_losses = np.linalg.solve(transition, squared_loss.T).T
    selected = corrected_losses[np.arange(len(labels)), labels.astype(int) - 1]
    return {"available": True, "n": len(selected),
            "estimated_latent_mse": float(selected.mean()),
            "standard_error": float(selected.std(ddof=1) / np.sqrt(len(selected)))
                              if len(selected) > 1 else None,
            "condition_number": condition,
            "interpretation": "Conditional noise-model estimate; may be negative; no square root. "
                              "Not verified clean-label risk or leaderboard RMSE."}


def report_split(predictions, calibration, transition, transform):
    candidates = {"observed_raw": predictions.observed,
                  "observed_calibrated": apply_shrinkage(predictions.observed, calibration)}
    latent, method = latent_candidate(predictions, transform)
    if latent is not None:
        candidates["latent_hypothesis"] = latent
    report = {"latent_candidate_method": method, "candidates": {}}
    proxy_mask = predictions.title_star > 0
    for name, values in candidates.items():
        item = {"against_observed_rating": metrics(values, predictions.labels),
                "inverse_transition_risk_hypothesis": corrected_latent_risk(
                    values, predictions.labels, transition)}
        if proxy_mask.any():
            item["against_title_proxy_NOT_ground_truth"] = metrics(
                values[proxy_mask], predictions.title_star[proxy_mask])
        report["candidates"][name] = item
    report["title_proxy_warning"] = (
        "Title star is an imperfect input-derived proxy and can be used by the model; "
        "matching it is not independent clean-label accuracy. This subset is not representative.")
    return report


def align_to_sample(ids, values, sample_ids):
    ids, sample_ids = np.asarray(ids), np.asarray(sample_ids)
    values = np.asarray(values, dtype=np.float64)
    if len(ids) != len(set(ids.tolist())) or len(sample_ids) != len(set(sample_ids.tolist())):
        raise ValueError("Duplicate prediction or sample id")
    if len(ids) != len(values) or set(ids.tolist()) != set(sample_ids.tolist()):
        raise ValueError("Prediction and sample id sets differ")
    if not np.isfinite(values).all() or np.any((values < 1) | (values > 5)):
        raise ValueError("Submission predictions must be finite and within [1,5]")
    lookup = dict(zip(ids.tolist(), values.tolist()))
    return np.asarray([lookup[key] for key in sample_ids.tolist()])


def save_submission(path, sample_ids, values):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "rating"])
        writer.writerows((int(i), format(float(v), ".10f")) for i, v in zip(sample_ids, values))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {out}")
    cal = read_predictions(args.cal, labelled=True)
    test = read_predictions(args.test)
    evaluation = read_predictions(args.eval, labelled=True) if args.eval else None
    if evaluation is not None and set(cal.ids.tolist()) & set(evaluation.ids.tolist()):
        raise ValueError("Calibration and evaluation IDs overlap")
    with open(args.metadata, encoding="utf-8") as stream:
        metadata = json.load(stream)
    transition = validate_transition(metadata["transition"])
    transform = affine_transition(transition)
    sample_ids = read_sample(args.sample)
    calibration = fit_shrinkage(cal.observed, cal.labels)
    calibration["calibration_input_sha256"] = digest(args.cal)
    calibration["metadata_sha256"] = digest(args.metadata)
    calibration["affine_transition"] = transform
    calibration["transition_basis"] = metadata.get("transition_basis", "unspecified; inspect source")
    calibration["transition_unsupported_rows"] = metadata.get("transition_unsupported_rows", [])
    calibration["label_noise_warning"] = (
        "Transition was estimated from title proxies, not certified clean labels. "
        "Latent output assumes that mechanism generalizes; test labels may use another mechanism.")
    report = {"calibration_fit_NOT_independent_evaluation": report_split(
                  cal, calibration, transition, transform),
              "evaluation": report_split(evaluation, calibration, transition, transform)
                            if evaluation is not None else None,
              "evaluation_policy": "Evaluation does not fit parameters or select a candidate.",
              "inputs_sha256": {"cal": digest(args.cal), "test": digest(args.test),
                                "sample": digest(args.sample), "metadata": digest(args.metadata)}}
    if args.eval:
        report["inputs_sha256"]["eval"] = digest(args.eval)
    candidates = {"observed_raw": test.observed,
                  "observed_calibrated": apply_shrinkage(test.observed, calibration)}
    latent, method = latent_candidate(test, transform)
    report["test_latent_candidate_method"] = method
    if latent is not None:
        candidates["latent_hypothesis"] = latent
    # Validate every candidate before any output is written.
    aligned = {name: align_to_sample(test.ids, values, sample_ids)
               for name, values in candidates.items()}
    out.mkdir(parents=True, exist_ok=True)
    for name, values in aligned.items():
        save_submission(out / f"{name}.csv", sample_ids, values)
    for name, value in (("calibration.json", calibration), ("metrics.json", report)):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                                encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()), "s": calibration["s"],
                      "anchor": calibration["anchor"], "candidates": list(aligned)}, indent=2))
    return calibration, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal", required=True)
    parser.add_argument("--eval")
    parser.add_argument("--test", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
