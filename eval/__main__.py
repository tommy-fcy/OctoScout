"""CLI entry point for the eval framework.

Usage:
    python -m eval
    python -m eval --category api_changes
    python -m eval --model claude-sonnet-4-6
    python -m eval --model claude-haiku-4-5 --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from eval.runner import print_report, run_eval


def main():
    parser = argparse.ArgumentParser(
        description="OctoScout Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m eval                                    # Run all cases with default model
  python -m eval --category api_changes             # Run only API change cases
  python -m eval --model claude-sonnet-4-6    # Use a specific model
  python -m eval --model claude-haiku-4-5 --model claude-sonnet-4-6  # Compare models
  python -m eval --verbose                          # Show agent reasoning
""",
    )
    parser.add_argument(
        "--category", "-c",
        help="Only run cases from a specific category subdirectory (e.g. 'api_changes')",
    )
    parser.add_argument(
        "--model", "-m",
        action="append",
        dest="models",
        help="Model name(s) to evaluate. Can be repeated to compare models.",
    )
    parser.add_argument(
        "--provider", "-p",
        default=None,
        help="LLM provider: 'claude' or 'openai'",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed agent reasoning steps",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max concurrent eval cases (default: 2)",
    )

    args = parser.parse_args()

    models = args.models or [None]  # None = use config default

    reports = []
    for model in models:
        label = model or "default"
        if len(models) > 1:
            print(f"\n{'=' * 60}")
            print(f"  Evaluating: {label}")
            print(f"{'=' * 60}")

        report = asyncio.run(run_eval(
            category=args.category,
            model=model,
            provider=args.provider,
            verbose=args.verbose,
            max_concurrent=args.concurrency,
        ))
        reports.append(report)
        print_report(report)

    # Comparison summary when running multiple models
    if len(reports) > 1:
        print("\n" + "=" * 60)
        print("  MODEL COMPARISON")
        print("=" * 60)
        for report in reports:
            print(
                f"  {report.model:30s}  "
                f"Pass: {report.pass_rate:.0%}  "
                f"Avg: {report.avg_score:.2f}  "
                f"Latency: {report.avg_latency:.1f}s"
            )


if __name__ == "__main__":
    main()
