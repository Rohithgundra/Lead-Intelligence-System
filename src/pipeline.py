"""
Pipeline orchestrator for Lead Intelligence System.
Handles CSV ingestion, data sanitization, batch chunking, execution,
and report generation.
"""

import csv
import logging
import os
import time
from typing import List, Optional, Tuple

from src.models import Lead, LeadEvaluation
from src.llm_client import LLMBatchClient
from src.reporting import (
    calculate_analytics,
    generate_json_report,
    generate_csv_report,
    print_cli_summary
)

logger = logging.getLogger(__name__)


class LeadPipeline:
    """End-to-end pipeline for batch lead qualification and outreach generation."""

    def __init__(
        self,
        batch_size: int = 5,
        provider: str = "auto",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        reference_date: Optional[str] = None,
        dry_run: bool = False,
        verbose: bool = False
    ):
        self.batch_size = max(1, batch_size)
        self.reference_date = reference_date
        self.verbose = verbose
        self.client = LLMBatchClient(
            provider=provider,
            model=model,
            api_key=api_key,
            reference_date=reference_date,
            dry_run=dry_run
        )

    def load_leads_from_csv(self, file_path: str) -> Tuple[List[Lead], Optional[str]]:
        """
        Ingest leads from CSV file.
        Infers reference anchor date as the maximum valid date in the file
        if not explicitly supplied, ensuring historical test files work properly.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Lead file not found at: {file_path}")

        leads: List[Lead] = []
        max_date: Optional[str] = None

        # Try utf-8-sig first to strip BOM if present, fallback to latin-1
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    reader = csv.DictReader(f)
                    # Normalize field names
                    if reader.fieldnames:
                        clean_fieldnames = [fn.strip().lower() for fn in reader.fieldnames]
                    else:
                        clean_fieldnames = []

                    for idx, row in enumerate(reader, start=1):
                        normalized_row = {
                            k.strip().lower(): (v.strip() if v else "")
                            for k, v in row.items() if k
                        }
                        lead = Lead.from_row(normalized_row, lead_id=idx)
                        leads.append(lead)

                        # Check for max interaction date
                        if lead.last_interaction_date:
                            if max_date is None or lead.last_interaction_date > max_date:
                                max_date = lead.last_interaction_date
                break
            except UnicodeDecodeError:
                continue

        logger.info(f"Loaded {len(leads)} leads from {file_path}")
        return leads, max_date

    def run(
        self,
        input_csv: str,
        output_json: str = "output_report.json",
        output_csv: str = "output_report.csv"
    ) -> List[LeadEvaluation]:
        """Execute full batch qualification pipeline."""
        start_time = time.time()
        print(f"\n[*] Starting Lead Intelligence Pipeline...")
        print(f"[*] Ingesting leads from: {input_csv}")

        leads, detected_max_date = self.load_leads_from_csv(input_csv)
        if not leads:
            print("[!] Warning: No leads found in dataset.")
            return []

        # Determine reference anchor date
        anchor_date = self.reference_date or detected_max_date or "2024-01-20"
        self.client.set_reference_date(anchor_date)
        print(f"[*] Reference Anchor Date: {anchor_date} (for recency calculation)")
        print(f"[*] Engine Provider:      {self.client.provider} (Model: {self.client.model})")
        print(f"[*] Batch Size:           {self.batch_size} leads/request")

        # Process in batches
        all_evaluations: List[LeadEvaluation] = []
        total_batches = (len(leads) + self.batch_size - 1) // self.batch_size

        for b_idx in range(total_batches):
            batch_start = b_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(leads))
            batch_leads = leads[batch_start:batch_end]

            print(
                f"    -> Processing Batch {b_idx + 1}/{total_batches} "
                f"(Leads {batch_start + 1}-{batch_end})...",
                end="\r"
            )

            batch_results = self.client.process_batch(batch_leads)
            all_evaluations.extend(batch_results)

        print(f"\n[✓] Finished batch evaluation of {len(all_evaluations)} leads.")

        elapsed_time = time.time() - start_time

        # Calculate analytics & business impact
        analytics = calculate_analytics(all_evaluations, elapsed_time)

        # Generate reports
        json_file = generate_json_report(
            evaluations=all_evaluations,
            analytics=analytics,
            output_path=output_json,
            model_name=self.client.model,
            provider=self.client.provider
        )
        csv_file = generate_csv_report(
            evaluations=all_evaluations,
            output_path=output_csv
        )

        # CLI Display
        print_cli_summary(analytics, all_evaluations, json_file, csv_file)

        return all_evaluations
