"""
Main CLI entrypoint for the Lead Intelligence System.
Forward Deployed AI Engineering solution for automated B2B lead qualification.
"""

import argparse
import logging
import os
import sys

from src.pipeline import LeadPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lead Intelligence System - AI-driven B2B lead qualification and personalized outreach.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # File options
    parser.add_argument(
        "-i", "--input",
        dest="input_file",
        default="leads.csv",
        help="Path to input CSV file containing leads (e.g., leads.csv or docs/leads_full_scale_100.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_json",
        default="output_report.json",
        help="Path where the final JSON report should be saved."
    )
    parser.add_argument(
        "--csv-output",
        dest="output_csv",
        default="output_report.csv",
        help="Path where the tabular CSV report should be saved."
    )

    # Batching and performance options
    parser.add_argument(
        "-b", "--batch-size",
        dest="batch_size",
        type=int,
        default=5,
        help="Number of leads to evaluate per batch."
    )

    # Provider and model options
    parser.add_argument(
        "-p", "--provider",
        dest="provider",
        default="auto",
        choices=["auto", "gemini", "openai", "groq", "rule-based"],
        help="LLM provider to use, or 'rule-based' for deterministic qualification."
    )
    parser.add_argument(
        "-m", "--model",
        dest="model",
        default=None,
        help="Model name (e.g., gemini-1.5-flash, gpt-4o-mini, llama-3.1-8b-instant)."
    )
    parser.add_argument(
        "-k", "--api-key",
        dest="api_key",
        default=None,
        help="API Key for LLM provider (can also be set via GEMINI_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY)."
    )

    # Rubric parameters
    parser.add_argument(
        "--reference-date",
        dest="reference_date",
        default=None,
        help="Anchor date for interaction recency (YYYY-MM-DD). If omitted, auto-detected from dataset."
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Run deterministic rubric evaluation without making external API calls."
    )
    parser.add_argument(
        "-v", "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose debug logging."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Fallback to sample if default input does not exist
    input_path = args.input_file
    if not os.path.exists(input_path):
        if os.path.exists("docs/leads_full_scale_100.csv"):
            input_path = "docs/leads_full_scale_100.csv"
            print(f"[*] Input '{args.input_file}' not found. Falling back to default: {input_path}")
        elif os.path.exists("docs/leads_training.csv"):
            input_path = "docs/leads_training.csv"
            print(f"[*] Input '{args.input_file}' not found. Falling back to: {input_path}")
        else:
            print(f"[!] Error: File '{input_path}' not found.")
            sys.exit(1)

    # Instantiate and execute pipeline
    pipeline = LeadPipeline(
        batch_size=args.batch_size,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        reference_date=args.reference_date,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    try:
        pipeline.run(
            input_csv=input_path,
            output_json=args.output_json,
            output_csv=args.output_csv
        )
    except Exception as e:
        print(f"\n[!] Pipeline Execution Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
