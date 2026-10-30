"""
Unit test suite for Lead Intelligence System.
Uses Python's built-in unittest module (zero dependencies).
"""

import os
import unittest
import tempfile
import json
from datetime import datetime

from src.models import Lead, FactorScore, OutreachMessage, LeadEvaluation
from src.rubric import QualificationRubric
from src.llm_client import LLMBatchClient
from src.reporting import calculate_analytics, generate_json_report, generate_csv_report
from src.pipeline import LeadPipeline


class TestLeadModels(unittest.TestCase):
    """Test data sanitization and model edge cases."""

    def test_clean_row_parsing(self):
        row = {
            "name": "Alice Chen",
            "company": "CloudScale AI",
            "company_size": "50",
            "industry": "SaaS",
            "source": "Inbound demo request",
            "last_interaction_date": "2024-01-15"
        }
        lead = Lead.from_row(row, lead_id=1)
        self.assertEqual(lead.name, "Alice Chen")
        self.assertEqual(lead.company, "CloudScale AI")
        self.assertEqual(lead.company_size, 50)
        self.assertEqual(lead.industry, "SaaS")
        self.assertEqual(len(lead.data_quality_issues), 0)

    def test_missing_data_sanitization(self):
        row = {
            "name": "",
            "company": "NA",
            "company_size": "UNKNOWN",
            "industry": "Unknown Sector",
            "source": "",
            "last_interaction_date": "invalid-date"
        }
        lead = Lead.from_row(row, lead_id=2)
        self.assertEqual(lead.name, "Valued Leader")
        self.assertEqual(lead.company, "Your Organization")
        self.assertIsNone(lead.company_size)
        self.assertEqual(lead.industry, "Unknown")
        self.assertEqual(lead.source, "Direct Inbound")
        self.assertIsNone(lead.last_interaction_date)
        self.assertTrue(len(lead.data_quality_issues) >= 4)


class TestQualificationRubric(unittest.TestCase):
    """Test rubric factor scoring, thresholds, and disqualification."""

    def setUp(self):
        self.rubric = QualificationRubric(reference_date="2024-01-20")

    def test_high_fit_qualification(self):
        lead = Lead(
            lead_id=1,
            name="Alice",
            company="Tech Corp",
            company_size=500,
            industry="SaaS",
            source="Inbound demo request",
            last_interaction_date="2024-01-18"
        )
        res = self.rubric.evaluate(lead)
        self.assertEqual(res.decision, "QUALIFIED")
        self.assertEqual(res.priority, "P1_HIGH")
        self.assertGreaterEqual(res.total_score, 8.5)
        self.assertTrue(len(res.outreach_messages) >= 2)

    def test_stale_disqualification(self):
        lead = Lead(
            lead_id=2,
            name="Bob",
            company="Old Corp",
            company_size=200,
            industry="Finance",
            source="Inbound demo request",
            last_interaction_date="2022-01-01"  # > 2 years stale
        )
        res = self.rubric.evaluate(lead)
        self.assertEqual(res.decision, "DISQUALIFIED")
        self.assertIn("Stale interaction", res.primary_reason)

    def test_micro_company_disqualification(self):
        lead = Lead(
            lead_id=3,
            name="Charlie",
            company="Solo Shop",
            company_size=2,
            industry="Consulting",
            source="LinkedIn outreach",
            last_interaction_date="2024-01-19"
        )
        res = self.rubric.evaluate(lead)
        self.assertEqual(res.decision, "DISQUALIFIED")
        self.assertIn("Micro-organization", res.primary_reason)

    def test_enterprise_review_flag(self):
        lead = Lead(
            lead_id=4,
            name="Diana",
            company="Mega Enterprise",
            company_size=25000,
            industry="Software",
            source="Inbound demo request",
            last_interaction_date="2024-01-19"
        )
        res = self.rubric.evaluate(lead)
        self.assertTrue(res.needs_review)
        self.assertTrue(any("Strategic Enterprise" in r for r in res.review_reasons))


class TestReporting(unittest.TestCase):
    """Test report generation and analytics calculations."""

    def test_sdr_hours_calculation(self):
        rubric = QualificationRubric(reference_date="2024-01-20")
        lead = Lead(
            lead_id=1,
            name="Test",
            company="Test Inc",
            company_size=100,
            industry="SaaS",
            source="Inbound demo request",
            last_interaction_date="2024-01-18"
        )
        evals = [rubric.evaluate(lead)] * 60  # 60 leads
        analytics = calculate_analytics(evals, processing_time_sec=1.0)

        # 60 leads * 10 min = 600 min = 10 hours
        self.assertEqual(analytics["business_roi"]["manual_qualification_hours_required"], 10.0)
        self.assertAlmostEqual(analytics["business_roi"]["sdr_hours_saved"], 10.0, places=1)


class TestFullPipeline(unittest.TestCase):
    """Test full pipeline execution on testing dataset."""

    def test_pipeline_on_testing_csv(self):
        pipeline = LeadPipeline(batch_size=5, dry_run=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_in = os.path.join(tmpdir, "test_leads.csv")
            # Create a self-contained sample test CSV if docs/ not available
            if os.path.exists("docs/leads_testing.csv"):
                csv_in = "docs/leads_testing.csv"
                expected_count = 20
            else:
                with open(csv_in, "w", encoding="utf-8") as f:
                    f.write("name,company,company_size,industry,source,last_interaction_date\n")
                    f.write("Test A,Corp A,100,SaaS,Inbound demo request,2024-01-20\n")
                    f.write("Test B,Corp B,5,Retail,LinkedIn outreach,2023-01-01\n")
                expected_count = 2

            json_out = os.path.join(tmpdir, "out.json")
            csv_out = os.path.join(tmpdir, "out.csv")

            evaluations = pipeline.run(
                input_csv=csv_in,
                output_json=json_out,
                output_csv=csv_out
            )
            self.assertEqual(len(evaluations), expected_count)
            self.assertTrue(os.path.exists(json_out))
            self.assertTrue(os.path.exists(csv_out))

            # Verify JSON loadable
            with open(json_out, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertEqual(data["executive_summary"]["total_processed"], expected_count)


if __name__ == "__main__":
    unittest.main()
