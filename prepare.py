#!/usr/bin/env python3
"""Strict CSV -> deterministic splits -> cross-fitted features using Spark.

The training target never contributes to its own entity features, including
the global smoothing prior. A title saying "Five Stars" is an imperfect proxy,
not a certified clean label. Its transition matrix is an experimental input.
Outputs use local/shared filesystem paths, not s3:// or hdfs:// URIs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time


ENTITY_SPECS = [("user", "user_id"), ("prod", "prod_id"), ("parent", "parent_prod_id")]
DENSE_COLUMNS = [
    "user_smooth", "user_log_count", "prod_smooth", "prod_log_count",
    "parent_smooth", "parent_log_count", "purchased_int", "votes_log",
    "votes_negative", "comment_empty", "title_length_log", "comment_length_log",
]


def transition_from_counts(counts, pseudocount=0.5):
    """Row = title cue 1..5, column = observed rating 1..5.

    An entirely unobserved row is identity, rather than a fictitious estimate
    of uniform noise. metadata records the counts and unsupported rows.
    """
    result = []
    for i, row in enumerate(counts):
        if sum(row) == 0:
            result.append([float(i == j) for j in range(5)])
        else:
            denom = sum(row) + 5 * pseudocount
            result.append([(v + pseudocount) / denom for v in row])
    return result


def fold_priors(total_count, total_sum, fold_totals, n_folds=5):
    """Training global prior also excludes the entire sample's internal fold."""
    return {
        k: ((total_sum - fold_totals.get(k, (0, 0.0))[1]) /
            (total_count - fold_totals.get(k, (0, 0.0))[0])
            if total_count > fold_totals.get(k, (0, 0.0))[0] else 3.0)
        for k in range(n_folds)
    }


def log(message):
    print(message, flush=True)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--master", default="local[8]")
    p.add_argument("--partitions", type=int, default=64)
    p.add_argument("--expected-train", type=int, default=7684645,
                   help="Use -1 for an intentionally different dataset.")
    p.add_argument("--expected-test", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--limit", type=int, default=0,
                   help="Prefix-only plumbing smoke test; disables expected row counts.")
    p.add_argument("--skip-retrieval", action="store_true",
                   help="Skip additional exact text/pair diagnostic jobs.")
    return p.parse_args()


def run(args):
    from pyspark import StorageLevel
    from pyspark.sql import SparkSession, functions as F, types as T

    if args.alpha <= 0 or args.partitions <= 0 or args.limit < 0:
        raise ValueError("alpha and partitions must be positive; limit must be nonnegative")
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output must be an empty/new directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_intermediate"
    tmp.mkdir()
    started = time.perf_counter()
    spark = (SparkSession.builder.appName("task2-crossfit-prepare")
             .master(args.master)
             .config("spark.sql.shuffle.partitions", str(args.partitions))
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.autoBroadcastJoinThreshold", "-1")
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    timings = {}

    def checkpoint(frame, name):
        path = str(tmp / name)
        frame.repartition(args.partitions).write.mode("errorifexists").parquet(path)
        return spark.read.parquet(path)

    def normalize_text(col):
        return F.trim(F.regexp_replace(F.coalesce(col, F.lit("")), r"\s+", " "))

    def read_input(path, labelled, name):
        fields = [T.StructField("id", T.LongType()), T.StructField("title", T.StringType()),
                  T.StructField("user_id", T.StringType()), T.StructField("comment", T.StringType()),
                  T.StructField("prod_id", T.StringType()), T.StructField("parent_prod_id", T.StringType()),
                  T.StructField("time", T.LongType()), T.StructField("votes", T.LongType()),
                  T.StructField("purchased", T.StringType())]
        if labelled:
            fields.append(T.StructField("rating", T.FloatType()))
        # Standard CSV double-quote escaping. Never use backslash as an escape
        # merely because the review text itself contains backslashes.
        df = (spark.read.schema(T.StructType(fields)).option("header", True)
              .option("enforceSchema", False).option("mode", "FAILFAST")
              .option("multiLine", True).option("quote", '"').option("escape", '"')
              .option("maxCharsPerColumn", -1).csv(path))
        if args.limit:
            df = df.limit(args.limit)
        if not labelled:
            df = df.withColumn("rating", F.lit(None).cast("float"))
        # Materializing ALL fields prevents count()/column-pruning from hiding
        # malformed fields during the FAILFAST integrity check.
        df = checkpoint(df, f"parsed_{name}")
        invalid = F.col("id").isNull() | F.col("time").isNull() | F.col("votes").isNull()
        for key in ("user_id", "prod_id", "parent_prod_id"):
            invalid = invalid | F.col(key).isNull() | (F.length(F.trim(F.col(key))) == 0)
        invalid = invalid | ~F.lower(F.col("purchased")).isin("true", "false") | F.col("purchased").isNull()
        if labelled:
            invalid = invalid | F.col("rating").isNull() | F.isnan("rating") | ~F.col("rating").isin(1., 2., 3., 4., 5.)
        summary = df.agg(F.count("*").alias("n"), F.countDistinct("id").alias("unique_ids"),
                         F.sum(F.when(invalid, 1).otherwise(0)).alias("invalid")).first()
        n = int(summary.n)
        expected = args.expected_train if labelled else args.expected_test
        if n == 0 or summary.unique_ids != n or summary.invalid:
            raise ValueError(f"{name}: invalid records/IDs: {summary.asDict()}")
        if not args.limit and expected >= 0 and n != expected:
            raise ValueError(f"{name}: found {n:,} rows; expected {expected:,}; inspect source/CSV dialect")
        df = df.withColumn("title", normalize_text(F.col("title")))
        df = df.withColumn("comment", normalize_text(F.col("comment")))
        # JSON preserves field boundaries; a delimiter inside a review cannot
        # merge otherwise different tuples. Rating and row ID are excluded.
        identity = F.to_json(F.struct("user_id", "prod_id", "time", "title", "comment"))
        df = df.withColumn("_group", F.sha2(identity, 256))
        bucket = F.conv(F.substring("_group", 1, 8), 16, 10).cast("long") % 10000
        fold = (F.conv(F.substring("_group", 9, 8), 16, 10).cast("long") % 5).cast("int")
        split = F.when(bucket < 8000, "train").when(bucket < 9000, "cal").otherwise("eval")
        df = df.withColumn("_split", split if labelled else F.lit("test")).withColumn("_fold", fold)
        star_map = F.create_map(*[F.lit(x) for pair in
                                 [("one star", 1), ("two stars", 2), ("three stars", 3),
                                  ("four stars", 4), ("five stars", 5)] for x in pair])
        df = df.withColumn("title_star", F.coalesce(star_map[F.lower("title")], F.lit(0)).cast("int"))
        return df, n

    try:
        t = time.perf_counter()
        raw_train, n_train = read_input(args.train, True, "train")
        raw_test, n_test = read_input(args.test, False, "test")
        base = checkpoint(raw_train.unionByName(raw_test), "base")
        timings["csv_parse_validate_split_seconds"] = time.perf_counter() - t
        counts = {r._split: int(r.n) for r in base.groupBy("_split").agg(F.count("*").alias("n")).collect()}
        if any(counts.get(s, 0) == 0 for s in ("train", "cal", "eval", "test")):
            raise ValueError(f"Empty partition: {counts}; use more records in --limit smoke tests")
        fit = base.where(F.col("_split") == "train")
        total = fit.agg(F.count("*").alias("n"), F.sum("rating").alias("s")).first()
        mu = float(total.s / total.n)
        fold_stats = {int(r._fold): (int(r.n), float(r.s)) for r in
                      fit.groupBy("_fold").agg(F.count("*").alias("n"), F.sum("rating").alias("s")).collect()}
        prior_values = fold_priors(int(total.n), float(total.s), fold_stats)
        prior_map = F.create_map(*[F.lit(x) for k, v in prior_values.items() for x in (k, v)])
        is_fit = F.col("_split") == "train"
        log(f"Validated rows: train CSV={n_train:,}; test CSV={n_test:,}; splits={counts}; fit mean={mu:.6f}")

        transition_counts = [[0] * 5 for _ in range(5)]
        for row in fit.where(F.col("title_star") > 0).groupBy("title_star", "rating").count().collect():
            transition_counts[int(row.title_star) - 1][int(row.rating) - 1] = int(row['count'])
        transition = transition_from_counts(transition_counts)

        t = time.perf_counter()
        featured = base
        for prefix, key in ENTITY_SPECS:
            entity_t = time.perf_counter()
            totals = (fit.groupBy(key).agg(F.count("*").alias("_n"), F.sum("rating").alias("_s"))
                      .persist(StorageLevel.DISK_ONLY))
            by_fold = (fit.groupBy(key, "_fold").agg(F.count("*").alias("_fn"), F.sum("rating").alias("_fs"))
                       .persist(StorageLevel.DISK_ONLY))
            joined = featured.join(totals, key, "left").join(by_fold, [key, "_fold"], "left")
            n = F.coalesce(F.col("_n"), F.lit(0)) - F.when(is_fit, F.coalesce(F.col("_fn"), F.lit(0))).otherwise(0)
            s = F.coalesce(F.col("_s"), F.lit(0.0)) - F.when(is_fit, F.coalesce(F.col("_fs"), F.lit(0.0))).otherwise(0.0)
            prior = F.when(is_fit, prior_map[F.col("_fold")]).otherwise(F.lit(mu))
            joined = joined.withColumn(prefix + "_smooth", (s + args.alpha * prior) / (n + args.alpha))
            joined = joined.withColumn(prefix + "_log_count", F.log1p(n.cast("double")))
            featured = checkpoint(joined.drop("_n", "_s", "_fn", "_fs"), "features_" + prefix)
            totals.unpersist()
            by_fold.unpersist()
            timings[prefix + "_crossfit_seconds"] = time.perf_counter() - entity_t
            log(f"Materialized {prefix} cross-fitted statistics")
        featured = (featured.withColumn("purchased_int", (F.lower("purchased") == "true").cast("float"))
                    .withColumn("votes_log", F.log1p(F.greatest(F.col("votes"), F.lit(0))))
                    .withColumn("votes_negative", (F.col("votes") < 0).cast("float"))
                    .withColumn("comment_empty", (F.length("comment") == 0).cast("float"))
                    .withColumn("title_length_log", F.log1p(F.length("title")))
                    .withColumn("comment_length_log", F.log1p(F.length("comment"))))
        final = featured.select("id", "rating", "title", "comment", "title_star", "_split",
                                F.array(*[F.col(c).cast("float") for c in DENSE_COLUMNS]).alias("dense"))
        for split in ("train", "cal", "eval", "test"):
            # Partition counts scale to the split, avoiding 64 tiny test shards.
            parts = min(args.partitions, max(1, math.ceil(counts[split] / 100000)))
            final.where(F.col("_split") == split).drop("_split").repartition(parts).write.mode("errorifexists").parquet(str(out / split))
        timings["crossfit_and_feature_output_seconds"] = time.perf_counter() - t

        diagnostics = {}
        if not args.skip_retrieval:
            t = time.perf_counter()
            targets = base.where(F.col("_split") != "train")
            for route in ("normalized_text", "user_product"):
                def route_key(frame):
                    if route == "normalized_text":
                        fields = F.struct(F.lower("title").alias("title"), F.lower("comment").alias("comment"))
                    else:
                        fields = F.struct("user_id", "prod_id")
                    return frame.withColumn("_key", F.sha2(F.to_json(fields), 256))
                idx = (route_key(fit).groupBy("_key").agg(F.count("*").alias("n"),
                       F.mean("rating").alias("mean"), F.var_pop("rating").alias("variance"),
                       F.countDistinct("rating").alias("label_values")).persist(StorageLevel.DISK_ONLY))
                index_summary = idx.agg(F.count("*").alias("groups"),
                    F.sum(F.when(F.col("n") > 1, 1).otherwise(0)).alias("repeated_groups"),
                    F.sum(F.when(F.col("label_values") > 1, 1).otherwise(0)).alias("conflicting_groups")).first().asDict()
                scored = route_key(targets).join(idx, "_key", "left")
                scored = (scored.withColumn("matched", F.col("n").isNotNull())
                          .withColumn("prediction", F.coalesce(F.col("mean"), F.lit(mu)))
                          .withColumn("smoothed_prediction", F.when(F.col("matched"),
                                      (F.col("n") * F.col("mean") + args.alpha * mu) / (F.col("n") + args.alpha)).otherwise(mu))
                          .withColumn("long_text", F.length("comment") >= 40))
                rows = scored.groupBy("_split").agg(F.count("*").alias("n_queries"),
                    F.sum(F.col("matched").cast("long")).alias("n_matched"),
                    F.sum(F.col("long_text").cast("long")).alias("n_long_queries"),
                    F.sum((F.col("matched") & F.col("long_text")).cast("long")).alias("n_long_matched"),
                    F.sqrt(F.mean((F.col("prediction") - F.col("rating")) ** 2)).alias("rmse_with_mean_fallback"),
                    F.sqrt(F.mean((F.col("smoothed_prediction") - F.col("rating")) ** 2)).alias("smoothed_rmse_with_mean_fallback"),
                    F.sqrt(F.mean(F.when(F.col("matched"), (F.col("prediction") - F.col("rating")) ** 2))).alias("matched_rmse"),
                    F.mean(F.when(F.col("matched"), F.col("variance"))).alias("mean_matched_label_variance"),
                    F.sum((F.col("label_values") > 1).cast("long")).alias("n_conflicting_matches")).collect()
                result = {r._split: {k: v for k, v in r.asDict().items() if k != "_split"} for r in rows}
                for val in result.values():
                    val["coverage"] = val["n_matched"] / val["n_queries"]
                    val["long_text_coverage"] = val["n_long_matched"] / val["n_long_queries"] if val["n_long_queries"] else None
                diagnostics[route] = {"index": index_summary, "queries": result,
                    "interpretation": "Attainable baseline with train-only index; not an oracle or RMSE bound. Same text need not mean same review."}
                idx.unpersist()
                log(f"Retrieval diagnostic {route}: {json.dumps(result)}")
            timings["retrieval_diagnostics_seconds"] = time.perf_counter() - t

        timings["total_prepare_seconds"] = time.perf_counter() - started
        metadata = {"schema_version": 2, "dense_columns": DENSE_COLUMNS, "label_mean": mu,
                    "transition": transition, "transition_counts": transition_counts,
                    "transition_unsupported_rows": [i + 1 for i, r in enumerate(transition_counts) if sum(r) == 0],
                    "transition_basis": "P(observed rating | exact title-star cue), fit split only; proxy assumption, not known clean labels",
                    "transition_pseudocount": 0.5, "counts": counts,
                    "input_rows": {"train": n_train, "test": n_test},
                    "split": {"method": "sha256 JSON(user_id, prod_id, time, normalized-whitespace title, comment)",
                              "bucket": "first 8 hex digits mod 10000; <8000 train, <9000 cal, else eval",
                              "internal_fold": "next 8 hex digits mod 5", "label_in_split_hash": False},
                    "target_encoding": {"method": "5-fold cross fit; sums and counts subtract internal fold; prior excludes same fold",
                                        "alpha": args.alpha, "fit_fold_priors": prior_values},
                    "smoke_prefix_limit": args.limit, "timing": timings,
                    "runtime": {"spark_version": spark.version, "master": spark.sparkContext.master,
                                "shuffle_partitions": args.partitions, "slurm_job_id": os.getenv("SLURM_JOB_ID")},
                    "retrieval_diagnostics": diagnostics}
        save_json(out / "metadata.json", metadata)
        save_json(out / "diagnostics.json", diagnostics)
        (out / "_PREPARE_SUCCESS").write_text("Complete; metadata and all four output splits exist.\n", encoding="utf-8")
        log(f"Preparation complete in {timings['total_prepare_seconds']:.1f}s: {out}")
        log(f"Intermediate checkpoints retained in {tmp}; they may be removed after verifying outputs.")
    finally:
        spark.stop()


if __name__ == "__main__":
    run(get_args())
