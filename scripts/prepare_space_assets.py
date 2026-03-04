from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REQUIRED_ARTIFACTS = ["results.duckdb", "config_resolved.yaml"]
OPTIONAL_ARTIFACTS = ["report.html", "report.md"]


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_preds(
    run_dir: Path,
    out_dir: Path,
    max_preds_bytes: int,
    max_sample_rows: int,
) -> None:
    preds = run_dir / "preds.jsonl"
    preds_sample = run_dir / "preds.sample.jsonl"

    if preds.exists():
        if preds.stat().st_size <= max_preds_bytes:
            _copy_file(preds, out_dir / "preds.jsonl")
            return

        out_sample = out_dir / "preds.sample.jsonl"
        out_sample.parent.mkdir(parents=True, exist_ok=True)
        with (
            preds.open("r", encoding="utf-8") as src_f,
            out_sample.open("w", encoding="utf-8") as dst_f,
        ):
            for idx, line in enumerate(src_f):
                if idx >= max_sample_rows:
                    break
                dst_f.write(line)
        return

    if preds_sample.exists():
        _copy_file(preds_sample, out_dir / "preds.sample.jsonl")


def prepare_experiment(
    exp_name: str,
    runs_root: Path,
    space_runs_root: Path,
    max_preds_bytes: int,
    max_sample_rows: int,
) -> None:
    run_dir = runs_root / exp_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory missing: {run_dir}")

    out_dir = space_runs_root / exp_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in REQUIRED_ARTIFACTS:
        src = run_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Required artifact missing: {src}")
        _copy_file(src, out_dir / name)

    for name in OPTIONAL_ARTIFACTS:
        src = run_dir / name
        if src.exists():
            _copy_file(src, out_dir / name)

    _copy_preds(run_dir, out_dir, max_preds_bytes=max_preds_bytes, max_sample_rows=max_sample_rows)
    print(f"Prepared space assets for '{exp_name}' -> {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare run artifacts for Hugging Face Space demo."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment names under runs/<experiment>/.",
    )
    parser.add_argument("--runs-root", default="runs", help="Local runs root directory.")
    parser.add_argument(
        "--space-runs-root",
        default="space_assets/runs",
        help="Destination runs root inside the repo for Space demo assets.",
    )
    parser.add_argument(
        "--max-preds-bytes",
        type=int,
        default=5_000_000,
        help="Copy full preds.jsonl only when at or under this size.",
    )
    parser.add_argument(
        "--max-sample-rows",
        type=int,
        default=200,
        help="Rows to keep when writing preds.sample.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    space_runs_root = Path(args.space_runs_root)
    for exp_name in args.experiments:
        prepare_experiment(
            exp_name=exp_name,
            runs_root=runs_root,
            space_runs_root=space_runs_root,
            max_preds_bytes=args.max_preds_bytes,
            max_sample_rows=args.max_sample_rows,
        )


if __name__ == "__main__":
    main()
