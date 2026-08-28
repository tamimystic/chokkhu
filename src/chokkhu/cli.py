from __future__ import annotations

import argparse
import sys
import chokkhu as ck


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chokkhu",
        description="Chokkhu: Unified End-to-End AI & Data Engineering Toolkit.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"chokkhu {ck.__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: eda
    eda_parser = subparsers.add_parser("eda", help="Run automated EDA")
    eda_parser.add_argument(
        "--data", "-d", required=True, help="Path to tabular or image dataset"
    )
    eda_parser.add_argument("--target", "-t", default=None, help="Target column name")
    eda_parser.add_argument(
        "--save-reports", action="store_true", help="Save visual reports to disk"
    )
    eda_parser.add_argument(
        "--save-dir", default=None, help="Directory to save EDA reports"
    )

    # Command: clean
    clean_parser = subparsers.add_parser(
        "clean", help="Clean missing values and outliers"
    )
    clean_parser.add_argument("--data", "-d", required=True, help="Path to dataset")
    clean_parser.add_argument(
        "--missing",
        default="median",
        choices=["mean", "median", "mode", "knn", "iterative", "drop"],
    )
    clean_parser.add_argument(
        "--outliers", default="iqr", choices=["iqr", "zscore", "isolation", "winsorize"]
    )
    clean_parser.add_argument(
        "--output", "-o", default=None, help="Output cleaned file path"
    )

    # Command: pipeline
    pipe_parser = subparsers.add_parser(
        "pipeline", help="Execute leak-free end-to-end ML pipeline"
    )
    pipe_parser.add_argument("--data", "-d", required=True, help="Path to dataset")
    pipe_parser.add_argument("--target", "-t", required=True, help="Target column name")
    pipe_parser.add_argument(
        "--model", "-m", default="random_forest", help="Model name"
    )
    pipe_parser.add_argument(
        "--task", default="auto", choices=["auto", "classification", "regression"]
    )
    pipe_parser.add_argument(
        "--save", "-s", default=None, help="Save pipeline to disk (.pkl)"
    )

    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 0

    if parsed_args.command == "eda":
        df = ck.load(parsed_args.data)
        ck.eda.tabular(
            df,
            target_col=parsed_args.target,
            save_reports=parsed_args.save_reports,
            save_dir=parsed_args.save_dir,
        )
    elif parsed_args.command == "clean":
        df = ck.load(parsed_args.data)
        cleaned = ck.clean(
            df, missing=parsed_args.missing, outliers=parsed_args.outliers
        )
        if parsed_args.output:
            ck.save(cleaned, parsed_args.output)
            print(f"Cleaned dataset saved to {parsed_args.output}")
        else:
            print(cleaned.head())
    elif parsed_args.command == "pipeline":
        df = ck.load(parsed_args.data)
        result = ck.pipeline(
            data=df,
            target=parsed_args.target,
            model=parsed_args.model,
            task=parsed_args.task,
        )
        print(result.summary())
        if parsed_args.save:
            result.save(parsed_args.save)

    return 0


if __name__ == "__main__":
    sys.exit(main())
