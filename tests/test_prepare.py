"""Pure statistical contracts; optional Spark smoke is exercised via prepare.py."""
import importlib.util
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location("prepare", Path(__file__).resolve().parents[1] / "prepare.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PrepareContracts(unittest.TestCase):
    def test_transition_orientation_and_missing_row(self):
        counts = [[80 if i == j else 5 for j in range(5)] for i in range(5)]
        counts[1] = [0] * 5
        transition = prepare.transition_from_counts(counts, pseudocount=0)
        self.assertEqual(transition[1], [0., 1., 0., 0., 0.])
        for row in transition:
            self.assertAlmostEqual(sum(row), 1.)
        self.assertAlmostEqual(sum((i + 1) * p for i, p in enumerate(transition[4])), 4.5)

    def test_global_prior_excludes_fold(self):
        # Singleton fold 0 contains rating 1; the other ratings are 4,5.
        priors = prepare.fold_priors(3, 10, {0: (1, 1), 1: (2, 9)})
        self.assertEqual(priors[0], 4.5)
        self.assertEqual(priors[1], 1.)
        # Change the excluded fold's label: its prior must be unchanged.
        changed = prepare.fold_priors(3, 14, {0: (1, 5), 1: (2, 9)})
        self.assertEqual(changed[0], priors[0])

    def test_no_other_folds_uses_fixed_prior(self):
        self.assertEqual(prepare.fold_priors(1, 5, {0: (1, 5)})[0], 3.)


@unittest.skipUnless(os.environ.get("RUN_SPARK_TESTS") == "1" and importlib.util.find_spec("pyspark"),
                     "Set RUN_SPARK_TESTS=1 with PySpark and a compatible Java installation")
class SparkIntegration(unittest.TestCase):
    def test_real_pipeline_crossfit_and_csv(self):
        from pyspark.sql import SparkSession
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cols = ["id", "title", "user_id", "comment", "prod_id", "parent_prod_id", "time", "votes", "purchased"]
            records = []
            for i in range(300):
                row = dict(zip(cols, [i, "Five Stars" if i % 3 else "Not good", f"u{i}",
                                     'A quoted "word", and \\ slash\nnext line' if i % 7 else "",
                                     f"p{i % 7}", f"a{i % 3}", 1600000000000 + i,
                                     -1 if i % 11 == 0 else i % 5, "True"]))
                row["rating"] = i % 5 + 1
                records.append(row)
            for name, labelled, subset in [("train", True, records), ("test", False, records[:20])]:
                with (root / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=cols + (["rating"] if labelled else []), extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(subset)
            args = SimpleNamespace(train=str(root / "train.csv"), test=str(root / "test.csv"),
                                   out=str(root / "prepared"), master="local[2]", partitions=2,
                                   expected_train=300, expected_test=20, alpha=20., limit=0,
                                   skip_retrieval=True)
            prepare.run(args)
            meta = json.loads((root / "prepared" / "metadata.json").read_text(encoding="utf-8"))
            expected = {}
            fit = []
            for raw in records:
                record = {k: (" ".join(raw[k].split()) if k in ["title", "comment"] else raw[k])
                          for k in ["user_id", "prod_id", "time", "title", "comment"]}
                digest = hashlib.sha256(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
                bucket = int(digest[:8], 16) % 10000
                split = "train" if bucket < 8000 else "cal" if bucket < 9000 else "eval"
                fold = int(digest[8:16], 16) % 5
                expected[raw["id"]] = (split, fold)
                if split == "train":
                    fit.append(raw)
            self.assertEqual(meta["counts"]["train"], len(fit))
            spark = SparkSession.builder.master("local[2]").appName("verify-crossfit").getOrCreate()
            try:
                for split in ["train", "cal", "eval"]:
                    for row in spark.read.parquet(str(root / "prepared" / split)).collect():
                        self.assertEqual(expected[row.id][0], split)
                        fold = expected[row.id][1]
                        refs = [r for r in fit if split != "train" or expected[r["id"]][1] != fold]
                        prior = sum(r["rating"] for r in refs) / len(refs)
                        # Unique user IDs have no reference row outside own fold.
                        self.assertAlmostEqual(row.dense[0], prior, places=5)
                        self.assertEqual(row.dense[1], 0.)
                        prod = records[row.id]["prod_id"]
                        hist = [r["rating"] for r in refs if r["prod_id"] == prod]
                        smooth = (sum(hist) + 20 * prior) / (len(hist) + 20)
                        self.assertAlmostEqual(row.dense[2], smooth, places=5)
                        self.assertEqual(len(row.dense), len(prepare.DENSE_COLUMNS))
            finally:
                spark.stop()


if __name__ == "__main__":
    unittest.main()
