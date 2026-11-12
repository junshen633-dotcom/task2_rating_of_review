#!/usr/bin/env python3
"""Collect four prespecified cal-only candidates and freeze a recommendation.

This script never reads final-evaluation predictions/metrics or leaderboard
scores. The selected conditional risk still depends on the estimated noise T.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from postprocess import (align_to_sample, digest, numeric, read_rows, read_sample,
                         save_submission, unique_ids)


BASELINE = "E4_seed43"
CANDIDATES = [
    ("ensemble_E4_two_seed", "E4_two_seed", ["E4", "E4_seed43"], "e4_two_seed_latent.csv"),
    ("ensemble_E34_four_model", "E34_four_model", ["E3", "E4", "E3_seed43", "E4_seed43"],
     "e34_four_model_latent.csv"),
    ("submission_E4_long8_single", "E4_long8_single", ["E4_long8_seed43_latent"],
     "e4_long8_single_latent.csv"),
    ("ensemble_E4_long8_plus_seed42", "E4_long8_plus_seed42", ["E4", "E4_long8_seed43_latent"],
     "e4_long8_plus_seed42_latent.csv"),
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validated_submission(path: Path, sample_ids):
    fields, rows = read_rows(path)
    if fields != ["id", "rating"]:
        raise ValueError(f"Submission must have exactly columns id,rating: {path}")
    ids = unique_ids(rows, path)
    values = np.asarray([numeric(row["rating"], "rating", 1, 5) for row in rows])
    return align_to_sample(ids, values, sample_ids)


def summary_row(name, models, weights, scores, filename, baseline=False):
    risk = scores["conditional_corrected_latent_risk"]
    if not risk.get("available"):
        raise ValueError(f"Corrected risk unavailable for {name}")
    estimate = numeric(risk["estimated_latent_mse"], "estimated_latent_mse")
    observed = scores["observed_against_observed_rating"]
    if int(risk["n"]) != int(observed["n"]):
        raise ValueError(f"Metric row counts differ for {name}")
    return {
        "name": name, "models": models, "weights": weights, "is_baseline": baseline,
        "submission_file": filename, "n_cal": int(risk["n"]),
        "observed_rmse": numeric(observed["rmse"], "observed_rmse", 0),
        "observed_mae": numeric(observed["mae"], "observed_mae", 0),
        "conditional_corrected_latent_mse": estimate,
        "conditional_corrected_latent_mse_standard_error": risk.get("standard_error"),
        "paired_corrected_mse_vs_baseline": scores["paired_corrected_mse_vs_baseline"],
        "paired_observed_mse_vs_baseline": scores["paired_observed_mse_vs_baseline"],
    }


def training_summary(work: Path, baseline_metadata_hash: str):
    source = work / "runs" / BASELINE / "metrics.json"
    extended_source = work / "runs/E4_long8_seed43/metrics.json"
    old, new = read_json(source), read_json(extended_source)
    for name, record in ((BASELINE, old), ("E4_long8_seed43", new)):
        if record.get("smoke") is not False or record.get("evaluation_split_used_for_selection") is not False:
            raise ValueError(f"Non-formal or non-cal-only training record: {name}")
        if record.get("metadata_sha256") != baseline_metadata_hash:
            raise ValueError(f"Training metadata do not match prediction metadata: {name}")
    old_epochs = {entry["epoch"]: entry for entry in old["epochs"]}
    new_epochs = {entry["epoch"]: entry for entry in new["epochs"]}
    prefix = []
    for epoch in (1, 2, 3):
        if epoch not in old_epochs or epoch not in new_epochs:
            raise ValueError("Missing first-three-epoch comparison data")
        before, after = old_epochs[epoch], new_epochs[epoch]
        old_rmse = before["calibration"]["rmse_observed"]
        new_rmse = after["calibration"]["rmse_observed"]
        prefix.append({"epoch": epoch, "baseline_observed_rmse": old_rmse,
                       "rerun_observed_rmse": new_rmse, "rmse_difference": new_rmse-old_rmse,
                       "baseline_train_loss": before["train_loss"],
                       "rerun_train_loss": after["train_loss"],
                       "train_loss_difference": after["train_loss"]-before["train_loss"],
                       "training_record_count_equal": before["train_records"] == after["train_records"],
                       "rerun_learning_rate": after.get("learning_rate")})
    best_observed = new["best_epoch"]
    best_latent = new["best_latent_epoch"]
    if best_observed not in new_epochs or best_latent not in new_epochs:
        raise ValueError("Selected best epochs are missing from the training history")
    history = [{"epoch": e["epoch"], "learning_rate": e.get("learning_rate"),
                "train_records": e["train_records"], "train_loss": e["train_loss"],
                "train_seconds": e["train_seconds"],
                "observed_cal_rmse": e["calibration"]["rmse_observed"],
                "conditional_corrected_latent_risk": e["calibration"].get("corrected_latent_risk")}
               for e in new["epochs"]]
    return {"experiment": "E4_long8_seed43", "method": "Rerun from initialization, then extend budget; not optimizer resume",
            "seed": new["seed"], "completed_epochs": len(history),
            "best_observed_epoch": best_observed, "best_observed_cal_rmse": new["best_cal_rmse"],
            "best_latent_epoch": best_latent,
            "best_conditional_corrected_latent_mse": new["best_cal_corrected_latent_mse"],
            "epochs": history, "first_three_epochs_vs_baseline": prefix,
            "architecture_equal_to_baseline": old["architecture"] == new["architecture"],
            "seed_equal_to_baseline": old["seed"] == new["seed"],
            "first_three_rmse_max_absolute_difference": max(abs(row["rmse_difference"]) for row in prefix),
            "baseline_metrics_sha256": digest(source), "extended_metrics_sha256": digest(extended_source)}, source, extended_source


def render_readme(selection):
    chosen = selection["chosen"]
    lines = ["# 扩展实验提交文件", "",
             f"首选文件：`recommended.csv`，对应 **{chosen['name']}**。", "",
             "推荐完全依据 cal 上的噪声修正潜在 MSE，候选权重预先固定为等权；没有使用 eval 或榜单分数选择。",
             "修正 MSE 是依赖噪声矩阵正确性的条件估计，不是已认证的干净标签误差；没有将它开平方冒充 RMSE。", "",
             "| 候选 | cal 观察 RMSE | 条件修正潜在 MSE | 文件 |",
             "|---|---:|---:|---|"]
    for row in selection["candidates"]:
        lines.append(f"| {row['name']} | {row['observed_rmse']:.8f} | "
                     f"{row['conditional_corrected_latent_mse']:.8f} | `{row['submission_file']}` |")
    training = selection["extended_training"]
    lines += ["", f"长训练完成 {training['completed_epochs']} 轮；观察 RMSE 最佳在第 "
              f"{training['best_observed_epoch']} 轮，修正潜在 MSE 最佳在第 {training['best_latent_epoch']} 轮。",
              f"前三轮相对原 E4_seed43 的 cal RMSE 最大绝对差：{training['first_three_rmse_max_absolute_difference']:.12g}。",
              "", "每份预测包含 10,000 条记录，已按 sampleSubmission 的 ID 对齐，并验证唯一性与评分 [1,5] 范围。",
              "`selection.json` 包含选择规则、全部比较、配对误差区间与训练记录；`analyses/` 保存各候选原始诊断。",
              "配对区间是固定噪声矩阵下按独立行近似的结果，未覆盖噪声矩阵估计、分组相关及模型选择带来的不确定性。",
              "已有基线文件保留为 `baseline_e4_seed43_latent.csv`；新候选不保证在 Kaggle 上优于基线。",
              "`SHA256SUMS` 可用于核对所有交付文件。", ""]
    return "\n".join(lines)


def run(args):
    work, out, sample = Path(args.work), Path(args.out), Path(args.sample)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {out}")
    sample_ids = read_sample(sample)
    if len(sample_ids) != 10000:
        raise ValueError(f"Expected 10000 sample IDs, found {len(sample_ids)}")
    sample_hash = digest(sample)
    analyses, rows, submissions, sources = {}, [], {}, {}
    metadata_hash = None
    baseline_cal_hash = None
    baseline_score = None
    for dirname, name, models, filename in CANDIDATES:
        path = work / dirname / "analysis.json"
        analysis = read_json(path)
        if analysis["name"] != name or analysis["ensemble_members"] != models:
            raise ValueError(f"Prespecified candidate identity mismatch: {dirname}")
        weights = analysis["weights"]
        if len(weights) != len(models) or not np.allclose(weights, [1./len(models)]*len(models), rtol=0, atol=1e-15):
            raise ValueError(f"Candidate weights are not prespecified equal weights: {dirname}")
        if analysis.get("baseline") != BASELINE or analysis["sample_sha256"] != sample_hash:
            raise ValueError(f"Baseline or sample mismatch for {name}")
        if analysis["test_records"] != 10000:
            raise ValueError(f"Test record count differs for {name}")
        current_baseline = analysis["members"][BASELINE]
        current_cal_hash = analysis["member_provenance"][BASELINE]["cal_predictions_sha256"]
        if metadata_hash is None:
            metadata_hash = analysis["metadata_sha256"]
            baseline_cal_hash = current_cal_hash
            baseline_score = current_baseline
            rows.append(summary_row(BASELINE, [BASELINE], [1.], baseline_score,
                                    "baseline_e4_seed43_latent.csv", baseline=True))
        elif (metadata_hash != analysis["metadata_sha256"] or baseline_cal_hash != current_cal_hash
              or baseline_score != current_baseline):
            raise ValueError(f"Candidate reports do not share one identical baseline: {name}")
        rows.append(summary_row(name, models, weights, analysis["ensemble"], filename))
        source = work / dirname / "latent_hypothesis.csv"
        submissions[filename] = validated_submission(source, sample_ids)
        analyses[dirname] = (path, analysis)
        sources[str(source.relative_to(work))] = digest(source)
        sources[str(path.relative_to(work))] = digest(path)
    if len({row["n_cal"] for row in rows}) != 1:
        raise ValueError("Candidates have inconsistent calibration row counts")
    baseline_path = work / "submission_E4_seed43/latent_hypothesis.csv"
    submissions["baseline_e4_seed43_latent.csv"] = validated_submission(baseline_path, sample_ids)
    sources[str(baseline_path.relative_to(work))] = digest(baseline_path)
    training, old_path, new_path = training_summary(work, metadata_hash)
    # Stable tie break keeps the incumbent when risk estimates are exactly equal.
    chosen = min(rows, key=lambda row: row["conditional_corrected_latent_mse"])
    selection = {
        "chosen": chosen, "recommended_file": "recommended.csv",
        "selected_name": chosen["name"], "selected_models": chosen["models"],
        "selected_weights": chosen["weights"],
        "selected_corrected_latent_mse": chosen["conditional_corrected_latent_mse"],
        "selected_candidate_file": chosen["submission_file"],
        "submission_files": {row["name"]: row["submission_file"] for row in rows},
        "selection_split": "cal", "evaluation_read": False, "leaderboard_used_for_selection": False,
        "policy": "Lowest conditional corrected latent MSE among baseline and four prespecified candidates; "
                  "equal ensemble weights; exact ties keep the earlier candidate (baseline first)",
        "candidates": rows, "extended_training": training,
        "metadata_sha256": metadata_hash, "sample_sha256": sample_hash,
        "source_artifacts_sha256": sources,
        "risk_warning": "Conditional on the estimated class-conditional feature-independent noise matrix. "
                        "Not measured clean-label error; uncertainty intervals exclude T estimation and selection.",
        "kaggle_submitted": False,
    }
    encoded = json.dumps(selection, indent=2, ensure_ascii=False, allow_nan=False)
    # All inputs, schemas and predictions were validated before creating outputs.
    out.mkdir(parents=True, exist_ok=True)
    for filename, values in submissions.items():
        save_submission(out / filename, sample_ids, values)
    save_submission(out / "recommended.csv", sample_ids, submissions[chosen["submission_file"]])
    (out / "selection.json").write_text(encoded, encoding="utf-8")
    (out / "README_CN.md").write_text(render_readme(selection), encoding="utf-8")
    for dirname, (path, _) in analyses.items():
        destination = out / "analyses" / dirname
        destination.mkdir(parents=True)
        shutil.copyfile(path, destination / "analysis.json")
    shutil.copyfile(old_path, out / "training_E4_seed43_baseline.json")
    shutil.copyfile(new_path, out / "training_E4_long8_seed43.json")
    checksum_lines = [f"{digest(path)}  {path.relative_to(out).as_posix()}"
                      for path in sorted(out.rglob("*")) if path.is_file()]
    (out / "SHA256SUMS").write_text("\n".join(checksum_lines)+"\n", encoding="utf-8")
    print(json.dumps({"chosen": chosen, "output": str(out.resolve()), "rows": len(sample_ids),
                      "completed_epochs": training["completed_epochs"]}, indent=2))
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
