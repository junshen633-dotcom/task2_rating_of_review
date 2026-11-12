"""Train, validate, and reproduce predictions from prepared Parquet shards.

Only calibration data choose the epoch. The held-out evaluation split is read
only when --evaluate-final is supplied, after loading the best checkpoint.
Spark preprocessing is separate; this script performs single-process CPU/GPU
training. Do not launch it with a multi-rank torchrun or duplicate SLURM tasks.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from model import ReviewModel, tokenize_hash


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class ParquetReviews(IterableDataset):
    """Assign each row group to exactly one worker, then bounded-buffer shuffle.

    Row-group offsets are assigned before shuffling. The explicit smoke limit
    therefore selects the same records regardless of worker count or epoch.
    """
    def __init__(self, directory: Path, *, dense_dim: int, shuffle: bool = False,
                 seed: int = 42, epoch: int = 0, buffer_size: int = 4096,
                 max_rows: int | None = None):
        super().__init__()
        import pyarrow.parquet as pq
        self.dense_dim, self.shuffle = dense_dim, shuffle
        self.seed, self.epoch, self.buffer_size = seed, epoch, buffer_size
        self.max_rows = max_rows
        self.row_groups = []
        offset = 0
        for filename in sorted(directory.glob("*.parquet")):
            metadata = pq.ParquetFile(filename).metadata
            for group in range(metadata.num_row_groups):
                rows = metadata.row_group(group).num_rows
                self.row_groups.append((str(filename), group, rows, offset))
                offset += rows
        if offset == 0:
            raise ValueError(f"No Parquet records found in {directory}")
        self.row_count = min(offset, max_rows) if max_rows else offset

    def __iter__(self):
        import pyarrow.parquet as pq
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        groups = list(self.row_groups)
        # All workers construct the same permutation before disjoint slicing.
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
        groups = groups[worker_id::workers]
        rng = random.Random(self.seed + self.epoch * 100003 + worker_id)
        columns = ["id", "rating", "title", "comment", "title_star"]
        if self.dense_dim:
            columns.append("dense")

        def records():
            for filename, group, _, offset in groups:
                if self.max_rows is not None and offset >= self.max_rows:
                    continue
                parquet = pq.ParquetFile(filename)
                seen = 0
                for batch in parquet.iter_batches(batch_size=512, row_groups=[group], columns=columns):
                    for record in batch.to_pylist():
                        position = offset + seen
                        seen += 1
                        if self.max_rows is not None and position >= self.max_rows:
                            break
                        dense = record.get("dense") or []
                        if len(dense) != self.dense_dim or not all(math.isfinite(v) for v in dense):
                            raise ValueError(f"Invalid dense features at id={record['id']}")
                        record["dense"] = dense
                        record["rating"] = (float(record["rating"]) if record["rating"] is not None
                                            else float("nan"))
                        star = int(record["title_star"] or 0)
                        if star < 0 or star > 5:
                            raise ValueError("title_star must be 0..5")
                        record["title_star"] = star
                        yield record

        if not self.shuffle:
            yield from records()
            return
        buffer = []
        for record in records():
            if len(buffer) < self.buffer_size:
                buffer.append(record)
            else:
                slot = rng.randrange(len(buffer))
                yield buffer[slot]
                buffer[slot] = record
        rng.shuffle(buffer)
        yield from buffer


class CollateReviews:
    def __init__(self, config: dict[str, Any], tokenizer=None):
        self.config, self.tokenizer = config, tokenizer

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if self.config["backbone"] == "hashcnn":
            ids = torch.tensor([tokenize_hash(r["title"], r["comment"], self.config["vocab_size"],
                                              self.config["max_tokens"]) for r in records], dtype=torch.long)
            attention = (ids != 0).long()
        else:
            # Tokenizer pair encoding inserts the backbone's own separator token.
            tokenized = self.tokenizer([r["title"] or "" for r in records],
                                      [r["comment"] or "" for r in records], padding=True,
                                      truncation=True, max_length=self.config["max_tokens"],
                                      return_tensors="pt")
            ids, attention = tokenized["input_ids"], tokenized["attention_mask"]
        return {"input_ids": ids, "attention_mask": attention,
                "dense": torch.tensor([r["dense"] for r in records], dtype=torch.float32),
                "rating": torch.tensor([r["rating"] for r in records], dtype=torch.float32),
                "title_star": [r["title_star"] for r in records], "id": [r["id"] for r in records]}


def make_loader(data_dir: Path, split: str, config: dict[str, Any], args, tokenizer,
                *, epoch: int = 0, training: bool = False) -> DataLoader:
    dataset = ParquetReviews(data_dir / split, dense_dim=config.get("dense_dim", 0),
                             shuffle=training, seed=args.seed, epoch=epoch,
                             buffer_size=args.shuffle_buffer,
                             max_rows=args.max_train_rows if training else None)
    generator = torch.Generator().manual_seed(args.seed + epoch)
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                      collate_fn=CollateReviews(config, tokenizer), generator=generator,
                      pin_memory=args.device.startswith("cuda"), drop_last=False,
                      persistent_workers=False)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def autocast_context(device: str, enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled and device.startswith("cuda") \
        else contextlib.nullcontext()


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def predict(model: ReviewModel, loader: DataLoader, device: str, amp: bool,
            destination: Path | None = None, *, require_labels: bool = True) -> dict[str, Any]:
    synchronize(device)
    started = time.perf_counter()
    model.eval()
    count, labelled, squared_error, loss_sum = 0, 0, 0., 0.
    handle = destination.open("w", newline="", encoding="utf-8") if destination else None
    writer = csv.writer(handle) if handle else None
    if writer:
        writer.writerow(["id", "rating", "title_star", "p_observed", "p_latent"])
    try:
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch(batch, device)
                with autocast_context(device, amp):
                    output = model(batch["input_ids"], batch["attention_mask"], batch["dense"])
                observed, latent = model.predictions(output)
                valid = torch.isfinite(batch["rating"])
                if require_labels and not valid.all():
                    raise ValueError("Missing labels in calibration/evaluation split")
                if valid.any():
                    n = int(valid.sum())
                    labels = batch["rating"][valid]
                    squared_error += ((observed[valid].double() - labels.double()) ** 2).sum().item()
                    loss_sum += model.loss(output[valid], labels).item() * n
                    labelled += n
                predictions, latent_predictions = observed.cpu().tolist(), latent.cpu().tolist()
                labels = batch["rating"].cpu().tolist()
                if writer:
                    for i, record_id in enumerate(batch["id"]):
                        writer.writerow([record_id, labels[i] if math.isfinite(labels[i]) else "",
                                         batch["title_star"][i], predictions[i],
                                         latent_predictions[i] if math.isfinite(latent_predictions[i]) else ""])
                count += len(predictions)
    finally:
        if handle:
            handle.close()
    synchronize(device)
    if count != loader.dataset.row_count:
        raise RuntimeError(f"Prediction coverage mismatch: {count} != {loader.dataset.row_count}")
    return {"records": count, "labelled_records": labelled,
            "rmse_observed": math.sqrt(squared_error / labelled) if labelled else None,
            "loss": loss_sum / labelled if labelled else None,
            "inference_seconds": time.perf_counter() - started,
            "includes_parquet_read_tokenization_and_csv_write": destination is not None}


def hardware_info(device: str) -> dict[str, Any]:
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {"platform": platform.platform(), "cpu_model": cpu_model,
            "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "servers": 1, "training_processes": 1, "device": device,
            "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "cuda_version": torch.version.cuda}


def save_checkpoint(path: Path, model: ReviewModel, **extras) -> None:
    payload = {"format_version": 1, "architecture": model.config,
               "transition": model.transition.detach().cpu(),
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, **extras}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[ReviewModel, dict[str, Any]]:
    # We save only primitive metadata and tensors, so no arbitrary-object pickle
    # loading is necessary. Requires torch >=2.1 (weights_only supported).
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = ReviewModel(payload["architecture"], payload["transition"], pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    return model, payload


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def build_tokenizer(config, output_dir: Path, *, checkpoint: Path | None = None):
    if config["backbone"] != "hf":
        return None
    from transformers import AutoTokenizer
    if checkpoint:
        location = checkpoint.parent / "tokenizer"
        if not location.exists():
            raise ValueError(f"Checkpoint tokenizer missing: {location}; copy tokenizer/ with checkpoint")
        tokenizer = AutoTokenizer.from_pretrained(str(location), trust_remote_code=False,
                                                 local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=False,
                                                 local_files_only=config.get("offline", False))
        tokenizer.save_pretrained(output_dir / "tokenizer")
    return tokenizer


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objective", choices=["mse", "ce", "forward"], default="forward")
    parser.add_argument("--backbone", choices=["hashcnn", "hf"], default="hashcnn")
    parser.add_argument("--metadata", action="store_true", help="Include prepared dense metadata features")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--offline", action="store_true", help="HF cache only; never download weights/tokenizer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=None, help="Default 1e-3 hashcnn; 2e-5 hf")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--hash-buckets", type=int, default=131072)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--cnn-channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="CUDA float16 AMP; safely disabled on CPU")
    parser.add_argument("--evaluate-final", action="store_true", help="Evaluate fixed best checkpoint on eval once")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Mark output as a non-final smoke experiment")
    parser.add_argument("--max-train-rows", type=int, help="Smoke only: deterministic prefix row limit")
    args = parser.parse_args()
    if args.max_train_rows is not None and (not args.smoke or args.max_train_rows <= 0):
        parser.error("--max-train-rows requires --smoke and a positive value")
    if args.predict_only and args.checkpoint is None:
        parser.error("--predict-only requires --checkpoint")
    if args.checkpoint is not None and not args.predict_only:
        parser.error("--checkpoint is for --predict-only; resume training is not implemented")
    if min(args.batch_size, args.max_epochs, args.patience, args.threads, args.shuffle_buffer) < 1:
        parser.error("batch-size, max-epochs, patience, threads, shuffle-buffer must be positive")
    if args.workers < 0 or args.max_tokens < 8 or args.hash_buckets < 8:
        parser.error("workers >=0; max-tokens and hash-buckets >=8 required")
    if not 0 <= args.dropout < 1 or (args.learning_rate is not None and args.learning_rate <= 0):
        parser.error("dropout must be in [0,1); learning-rate must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        parser.error("This trainer uses one process. Request one SLURM task; do not use multi-rank torchrun.")
    return args


def main():
    started = time.perf_counter()
    args = get_args()
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "metrics.json").exists():
        raise FileExistsError("output-dir already has metrics.json; use a new directory to preserve experiments")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        requested_device = torch.device(args.device)
        torch.cuda.set_device(0 if requested_device.index is None else requested_device.index)
        torch.cuda.reset_peak_memory_stats()
    amp = args.amp and args.device.startswith("cuda")
    metadata_path = args.data_dir / "metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    metadata_digest = hashlib.sha256(metadata_bytes).hexdigest()
    if args.evaluate_final and (args.output_dir / "predictions_eval.csv").exists():
        raise FileExistsError("Final evaluation already exists; do not repeatedly select on eval")
    metrics = {"smoke": args.smoke, "seed": args.seed, "hardware": hardware_info(args.device),
               "metadata_sha256": metadata_digest, "epochs": [], "amp_enabled": amp,
               "row_limit": args.max_train_rows, "epoch_selection_split": "cal",
               "evaluation_split_used_for_selection": False,
               "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    if args.predict_only:
        model, checkpoint = load_checkpoint(args.checkpoint, args.device)
        if checkpoint.get("metadata_sha256") != metadata_digest:
            raise ValueError("Prepared metadata differ from checkpoint; reproduce the original preparation")
        config = model.config
        metrics["smoke"] = bool(checkpoint.get("smoke", False))
        tokenizer = build_tokenizer(config, args.output_dir, checkpoint=args.checkpoint)
        best_path = args.checkpoint
    else:
        if (args.output_dir / "best.pt").exists():
            raise FileExistsError("best.pt already exists; use a new output directory")
        config = {"objective": args.objective, "backbone": args.backbone,
                  "model_name": args.model_name, "offline": args.offline,
                  "vocab_size": args.hash_buckets, "max_tokens": args.max_tokens,
                  "embedding_dim": args.embedding_dim, "cnn_channels": args.cnn_channels,
                  "hidden_dim": args.hidden_dim, "dropout": args.dropout, "kernels": [3, 5],
                  "dense_dim": len(metadata["dense_columns"]) if args.metadata else 0,
                  "label_mean": float(metadata["label_mean"])}
        tokenizer = build_tokenizer(config, args.output_dir)
        model = ReviewModel(config, metadata["transition"]).to(args.device)
        config = model.config
        learning_rate = args.learning_rate or (2e-5 if args.backbone == "hf" else 1e-3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=amp)
        best_rmse, bad_epochs = float("inf"), 0
        best_path = args.output_dir / "best.pt"
        for epoch in range(1, args.max_epochs + 1):
            synchronize(args.device)
            epoch_started = time.perf_counter()
            loader = make_loader(args.data_dir, "train", config, args, tokenizer, epoch=epoch, training=True)
            model.train()
            total_loss, count = 0., 0
            for batch in loader:
                batch = move_batch(batch, args.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(args.device, amp):
                    output = model(batch["input_ids"], batch["attention_mask"], batch["dense"])
                    loss = model.loss(output, batch["rating"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                batch_count = len(batch["id"])
                total_loss += loss.detach().item() * batch_count
                count += batch_count
            synchronize(args.device)
            training_seconds = time.perf_counter() - epoch_started
            if count != loader.dataset.row_count:
                raise RuntimeError(f"Training coverage mismatch: {count} != {loader.dataset.row_count}")
            calibration = predict(model, make_loader(args.data_dir, "cal", config, args, tokenizer),
                                  args.device, amp)
            entry = {"epoch": epoch, "train_records": count, "train_loss": total_loss / count,
                     "train_seconds": training_seconds, "calibration": calibration}
            metrics["epochs"].append(entry)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            if calibration["rmse_observed"] < best_rmse:
                best_rmse, bad_epochs = calibration["rmse_observed"], 0
                save_checkpoint(best_path, model, epoch=epoch, cal_rmse=best_rmse, smoke=args.smoke,
                                metadata_sha256=metadata_digest, seed=args.seed)
                metrics["best_epoch"], metrics["best_cal_rmse"] = epoch, best_rmse
            else:
                bad_epochs += 1
            write_json(args.output_dir / "training_progress.json", metrics)
            if bad_epochs >= args.patience:
                break
        model, checkpoint = load_checkpoint(best_path, args.device)
    metrics["architecture"] = model.config
    metrics["checkpoint"] = str(best_path.resolve())
    metrics["best_epoch"] = checkpoint.get("epoch")
    metrics["best_cal_rmse"] = checkpoint.get("cal_rmse")
    synchronize(args.device)
    metrics["training_script_offline_seconds"] = time.perf_counter() - started if not args.predict_only else None
    preprocessing_seconds = metadata.get("preprocessing_seconds", metadata.get("timing", {}).get("total_prepare_seconds"))
    metrics["preprocessing_seconds"] = preprocessing_seconds
    metrics["total_offline_seconds"] = (preprocessing_seconds + metrics["training_script_offline_seconds"]
                                         if not args.predict_only and isinstance(preprocessing_seconds, (int, float))
                                         else None)
    metrics["offline_definition"] = "prepare preprocessing_seconds + training script through best checkpoint reload; excludes final prediction passes"
    metrics["predictions"] = {}
    for split in (["cal", "eval", "test"] if args.evaluate_final else ["cal", "test"]):
        prediction_started = time.perf_counter()
        loader = make_loader(args.data_dir, split, model.config, args, tokenizer)
        result = predict(model, loader, args.device, amp, args.output_dir / f"predictions_{split}.csv",
                         require_labels=split != "test")
        result["total_split_inference_seconds"] = time.perf_counter() - prediction_started
        metrics["predictions"][split] = result
        print(json.dumps({"split": split, **result}, ensure_ascii=False), flush=True)
    metrics["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    metrics["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else 0
    metrics["total_script_seconds"] = time.perf_counter() - started
    write_json(args.output_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
