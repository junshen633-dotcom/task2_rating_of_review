#!/usr/bin/env python3
"""Fixed equal-weight latent ensembles; development diagnostics, never final eval.

Noise-corrected risk is conditional on a correct, feature-independent transition
matrix. Its row-IID uncertainty ignores estimation of T and model selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from postprocess import (MAX_TRANSITION_CONDITION, Predictions, align_to_sample,
                         corrected_latent_risk, digest, metrics, read_predictions,
                         read_sample, save_submission, validate_transition)


UNCERTAINTY = ("Row-IID approximation conditional on fixed T; excludes group dependence, "
               "transition-estimation uncertainty and model-selection effects. "
               "Not a clean-label accuracy or leaderboard guarantee.")


def sorted_predictions(predictions: Predictions) -> Predictions:
    """Sort on IDs without assuming parquet/prediction row order is stable."""
    if predictions.latent is None:
        raise ValueError("Every member must contain finite p_latent predictions")
    order = np.argsort(predictions.ids)
    if len(np.unique(predictions.ids)) != len(order):
        raise ValueError("Duplicate prediction IDs")
    return Predictions(predictions.ids[order], predictions.observed[order],
                       predictions.title_star[order],
                       predictions.labels[order] if predictions.labels is not None else None,
                       predictions.latent[order])


def assert_aligned(reference: Predictions, current: Predictions) -> None:
    if not np.array_equal(reference.ids, current.ids):
        raise ValueError("Member prediction ID sets differ")
    if not np.array_equal(reference.title_star, current.title_star):
        raise ValueError("Member title-star values differ for matched IDs")
    if (reference.labels is None) != (current.labels is None):
        raise ValueError("Member label availability differs")
    if reference.labels is not None and not np.array_equal(reference.labels, current.labels):
        raise ValueError("Member labels differ for matched IDs")


def equal_weight(members: list[Predictions]) -> Predictions:
    if not members:
        raise ValueError("At least one ensemble member is required")
    members = [sorted_predictions(member) for member in members]
    reference = members[0]
    for member in members:
        assert_aligned(reference, member)
    return Predictions(reference.ids.copy(), np.mean([p.observed for p in members], axis=0),
                       reference.title_star.copy(),
                       reference.labels.copy() if reference.labels is not None else None,
                       np.mean([p.latent for p in members], axis=0))


def corrected_losses(predictions, labels, transition):
    """Per-row backward-corrected squared loss, via a five-row polynomial basis."""
    transition = validate_transition(transition)
    if np.linalg.cond(transition) > MAX_TRANSITION_CONDITION:
        raise ValueError("Transition is too ill-conditioned for corrected comparisons")
    p, y = np.asarray(predictions, dtype=np.float64), np.asarray(labels)
    if p.ndim != 1 or p.shape != y.shape or len(p) == 0:
        raise ValueError("Corrected loss inputs must be aligned nonempty vectors")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite corrected loss input")
    if np.any(y != np.floor(y)) or np.any((y < 1) | (y > 5)):
        raise ValueError("Observed labels must be integers in [1,5]")
    stars = np.arange(1., 6.)
    coefficients = np.linalg.solve(transition, np.column_stack([stars**2, stars, np.ones(5)]))
    selected = coefficients[y.astype(np.int64) - 1]
    return selected[:, 0] - 2 * p * selected[:, 1] + p*p * selected[:, 2]


def paired_difference(current, reference):
    difference = np.asarray(current) - np.asarray(reference)
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / np.sqrt(len(difference))) if len(difference) > 1 else None
    return {"n": len(difference), "mean_candidate_minus_baseline": mean,
            "standard_error": se,
            "approx_95pct_interval": [mean-1.96*se, mean+1.96*se] if se is not None else None,
            "negative_favors_candidate": True, "uncertainty": UNCERTAINTY}


def score(predictions: Predictions, transition, baseline: Predictions) -> dict:
    assert_aligned(baseline, predictions)
    y = predictions.labels
    if y is None:
        raise ValueError("Calibration labels required")
    result = {
        "observed_against_observed_rating": metrics(predictions.observed, y),
        "latent_against_observed_rating_NOT_clean_risk": metrics(predictions.latent, y),
        "conditional_corrected_latent_risk": corrected_latent_risk(predictions.latent, y, transition),
        "paired_corrected_mse_vs_baseline": paired_difference(
            corrected_losses(predictions.latent, y, transition),
            corrected_losses(baseline.latent, y, transition)),
        "paired_observed_mse_vs_baseline": paired_difference(
            (predictions.observed-y)**2, (baseline.observed-y)**2),
        "title_cue_subsets": {},
    }
    for name, mask in (("explicit_star_title", predictions.title_star > 0),
                       ("no_explicit_star_title", predictions.title_star == 0)):
        if not mask.any():
            result["title_cue_subsets"][name] = {"n": 0}
            continue
        subgroup = {"n": int(mask.sum()),
                    "observed_against_observed_rating": metrics(predictions.observed[mask], y[mask]),
                    "conditional_corrected_latent_risk": corrected_latent_risk(
                        predictions.latent[mask], y[mask], transition)}
        if name == "explicit_star_title":
            subgroup["latent_against_input_title_proxy_NOT_independent_ground_truth"] = metrics(
                predictions.latent[mask], predictions.title_star[mask])
        result["title_cue_subsets"][name] = subgroup
    return result


def write_predictions(path: Path, predictions: Predictions):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "rating", "title_star", "p_observed", "p_latent"])
        for i, record_id in enumerate(predictions.ids):
            writer.writerow([int(record_id), float(predictions.labels[i]) if predictions.labels is not None else "",
                             int(predictions.title_star[i]), float(predictions.observed[i]),
                             float(predictions.latent[i])])


def run(args):
    out, root = Path(args.out), Path(args.runs_root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {out}")
    names = list(args.models)
    if not names or len(set(names)) != len(names):
        raise ValueError("Ensemble model names must be nonempty and unique")
    baseline_name = getattr(args, "baseline", "E4_seed43")
    for name in names + [baseline_name]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Model names must be directory basenames using letters, digits, _ or -")
    metadata_hash = digest(args.metadata)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    transition = validate_transition(metadata["transition"])
    sample_ids = read_sample(args.sample)
    cal_members, test_members, manifests = {}, {}, {}
    for name in dict.fromkeys(names + [baseline_name]):
        folder = root / name
        member_metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        if member_metrics.get("smoke") is not False:
            raise ValueError(f"Refusing smoke or unverified member: {name}")
        if member_metrics.get("metadata_sha256") != metadata_hash:
            raise ValueError(f"Prepared metadata digest mismatch for member: {name}")
        if member_metrics.get("evaluation_split_used_for_selection") is not False:
            raise ValueError(f"Member lacks cal-only selection provenance: {name}")
        cal_path = folder / "predictions_cal.csv"
        cal_members[name] = sorted_predictions(read_predictions(cal_path, labelled=True))
        manifests[name] = {"metrics_sha256": digest(folder / "metrics.json"),
                           "cal_predictions_sha256": digest(cal_path),
                           "best_epoch": member_metrics.get("best_epoch"),
                           "seed": member_metrics.get("seed")}
        if name in names:
            test_path = folder / "predictions_test.csv"
            test_members[name] = sorted_predictions(read_predictions(test_path))
            manifests[name]["test_predictions_sha256"] = digest(test_path)
    baseline = cal_members[baseline_name]
    ensemble_cal = equal_weight([cal_members[n] for n in names])
    ensemble_test = equal_weight([test_members[n] for n in names])
    # Validate the full submission contract before writing anything.
    latent_submission = align_to_sample(ensemble_test.ids, ensemble_test.latent, sample_ids)
    observed_submission = align_to_sample(ensemble_test.ids, ensemble_test.observed, sample_ids)
    report = {
        "name": args.name, "scope": "Calibration/development only; final eval is not read",
        "ensemble_members": names, "weights": [1./len(names)] * len(names),
        "weight_policy": "Prespecified equal weights; not fitted to labels or leaderboard scores",
        "baseline": baseline_name, "metadata_sha256": metadata_hash,
        "sample_sha256": digest(args.sample), "member_provenance": manifests,
        "members": {n: score(p, transition, baseline) for n, p in cal_members.items()},
        "ensemble": score(ensemble_cal, transition, baseline),
        "test_records": len(sample_ids), "uncertainty": UNCERTAINTY,
        "transition_basis": metadata.get("transition_basis", "Inspect preparation metadata"),
        "title_proxy_warning": "Title cues are model inputs, not independent clean ground truth. "
                               "Subset corrected risk additionally assumes the same T applies in that subset.",
    }
    # A compact manifest permits subsequent fixed ensembles to reuse these outputs.
    manifest = {"smoke": False, "seed": None, "metadata_sha256": metadata_hash,
                "evaluation_split_used_for_selection": False, "best_epoch": None,
                "best_cal_rmse": report["ensemble"]["observed_against_observed_rating"]["rmse"],
                "ensemble_members": names, "weights": report["weights"]}
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    out.mkdir(parents=True, exist_ok=True)
    write_predictions(out / "predictions_cal.csv", ensemble_cal)
    write_predictions(out / "predictions_test.csv", ensemble_test)
    save_submission(out / "latent_hypothesis.csv", sample_ids, latent_submission)
    save_submission(out / "observed_raw.csv", sample_ids, observed_submission)
    (out / "analysis.json").write_text(encoded, encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"name": args.name, "members": names, "test_records": len(sample_ids),
                      "observed_rmse": manifest["best_cal_rmse"],
                      "corrected_latent_risk": report["ensemble"]["conditional_corrected_latent_risk"],
                      "paired_corrected_mse_vs_baseline": report["ensemble"]["paired_corrected_mse_vs_baseline"],
                      "output": str(out.resolve())}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline", default="E4_seed43")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
