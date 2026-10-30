"""
Reporting and Analytics module for Lead Intelligence System.
Generates comprehensive JSON and CSV reports, calculates business ROI
(SDR hours saved, pipeline acceleration), and formats rich CLI outputs.
"""

import csv
import json
import os
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any

from src.models import LeadEvaluation


def calculate_analytics(evaluations: List[LeadEvaluation], processing_time_sec: float) -> Dict[str, Any]:
    """Calculate aggregate statistics, distributions, and business ROI metrics."""
    total = len(evaluations)
    if total == 0:
        return {}

    qualified = [e for e in evaluations if e.decision == "QUALIFIED"]
    disqualified = [e for e in evaluations if e.decision == "DISQUALIFIED"]
    review = [e for e in evaluations if e.decision == "REVIEW"]

    p1_leads = [e for e in qualified if e.priority == "P1_HIGH"]
    p2_leads = [e for e in qualified if e.priority == "P2_MEDIUM"]

    # Calculate SDR time saved:
    # Baseline: Sales team manually qualifies leads at 8-12 minutes per decision (avg: 10 minutes).
    # Automated system takes processing_time_sec / total per lead (fractions of a second).
    manual_hours = (total * 10.0) / 60.0
    automated_hours = processing_time_sec / 3600.0
    hours_saved = max(0.0, manual_hours - automated_hours)

    # Industry distribution
    industry_counts = Counter(e.lead.industry for e in evaluations)
    qualified_industries = Counter(e.lead.industry for e in qualified)

    # Company size bracket distribution
    def get_size_bracket(size):
        if size is None:
            return "Unknown / NA"
        if size < 10:
            return "1-9 (Micro/Solopreneur)"
        if size < 50:
            return "10-49 (Small)"
        if size < 250:
            return "50-249 (Mid-market lower)"
        if size <= 2500:
            return "250-2500 (Mid-market sweet spot)"
        if size <= 10000:
            return "2501-10000 (Upper mid-market)"
        return "10000+ (Enterprise)"

    size_distribution = Counter(get_size_bracket(e.lead.company_size) for e in evaluations)

    # Source channel distribution
    source_distribution = Counter(e.lead.source for e in evaluations)

    # Common disqualification reasons
    disqualification_reasons = Counter()
    for e in disqualified:
        # Simplify reason to core category
        r = e.primary_reason.lower()
        if "stale" in r:
            category = "Stale interaction (> 180 days)"
        elif "micro" in r or "below icp" in r:
            category = "Micro-company / Solopreneur lack of budget"
        elif "missing" in r:
            category = "Missing critical contact or organization identity"
        elif "score" in r or "threshold" in r:
            category = "Low aggregate qualification score (< 5.0/10)"
        else:
            category = e.primary_reason[:60]
        disqualification_reasons[category] += 1

    return {
        "total_processed": total,
        "qualified_count": len(qualified),
        "qualified_pct": round((len(qualified) / total) * 100, 1),
        "review_count": len(review),
        "review_pct": round((len(review) / total) * 100, 1),
        "disqualified_count": len(disqualified),
        "disqualified_pct": round((len(disqualified) / total) * 100, 1),
        "rejected_count": len(disqualified),
        "rejected_pct": round((len(disqualified) / total) * 100, 1),
        "priority_distribution": {
            "P1_HIGH": len(p1_leads),
            "P2_MEDIUM": len(p2_leads),
            "unassigned_or_review": len(review) + len(disqualified)
        },
        "business_roi": {
            "manual_baseline_minutes_per_lead": 10.0,
            "manual_qualification_hours_required": round(manual_hours, 1),
            "automated_processing_time_seconds": round(processing_time_sec, 2),
            "sdr_hours_saved": round(hours_saved, 1),
            "time_saved_percentage": round((1.0 - (automated_hours / max(manual_hours, 0.001))) * 100, 2)
        },
        "industry_distribution": dict(industry_counts.most_common(10)),
        "qualified_by_industry": dict(qualified_industries.most_common(10)),
        "company_size_distribution": dict(size_distribution),
        "source_distribution": dict(source_distribution.most_common(10)),
        "common_disqualification_reasons": dict(disqualification_reasons.most_common(5)),
        "common_rejection_reasons": dict(disqualification_reasons.most_common(5))
    }


def generate_json_report(
    evaluations: List[LeadEvaluation],
    analytics: Dict[str, Any],
    output_path: str,
    model_name: str,
    provider: str
) -> str:
    """Write complete structured JSON report."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    qualified_sorted = sorted(
        [e for e in evaluations if e.decision == "QUALIFIED"],
        key=lambda x: x.total_score,
        reverse=True
    )
    review_queue = [e for e in evaluations if e.decision == "REVIEW"]
    disqualified_archive = [e for e in evaluations if e.decision == "DISQUALIFIED"]

    # Select 5 diverse, high-quality sample outreach messages
    sample_messages = []
    seen_industries = set()
    for e in qualified_sorted:
        if e.outreach_messages and e.lead.industry not in seen_industries:
            seen_industries.add(e.lead.industry)
            sample_messages.append({
                "lead_name": e.lead.name,
                "company": e.lead.company,
                "industry": e.lead.industry,
                "source": e.lead.source,
                "score": round(e.total_score, 1),
                "priority": e.priority,
                "primary_variant": e.outreach_messages[0].variant,
                "subject": e.outreach_messages[0].subject,
                "body": e.outreach_messages[0].body
            })
            if len(sample_messages) >= 5:
                break

    report_payload = {
        "metadata": {
            "report_title": "Lead Intelligence System - Qualification & Outreach Report",
            "generated_at": datetime.now().isoformat(),
            "model": model_name,
            "provider": provider,
            "system_version": "1.0.0"
        },
        "executive_summary": analytics,
        "priority_queue": [e.to_dict() for e in qualified_sorted],
        "priority_queue_qualified": [e.to_dict() for e in qualified_sorted],
        "flagged_leads": [e.to_dict() for e in review_queue],
        "review_queue_flagged": [e.to_dict() for e in review_queue],
        "disqualified_leads": [e.to_dict() for e in disqualified_archive],
        "rejected_leads": [e.to_dict() for e in disqualified_archive],
        "disqualified_archive": [e.to_dict() for e in disqualified_archive],
        "sample_outreach_messages": sample_messages,
        "all_evaluations": [e.to_dict() for e in evaluations]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    return output_path


def generate_csv_report(evaluations: List[LeadEvaluation], output_path: str) -> str:
    """Write tabular CSV export for CRM import."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fieldnames = [
        "lead_id",
        "name",
        "company",
        "company_size",
        "industry",
        "source",
        "last_interaction_date",
        "qualification_score",
        "decision",
        "priority",
        "primary_reason",
        "needs_review",
        "review_reasons",
        "data_quality_issues",
        "outreach_subject_variant_1",
        "outreach_body_variant_1"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for e in evaluations:
            first_msg = e.outreach_messages[0] if e.outreach_messages else None
            writer.writerow({
                "lead_id": e.lead_id,
                "name": e.lead.name,
                "company": e.lead.company,
                "company_size": e.lead.company_size if e.lead.company_size is not None else "Unknown",
                "industry": e.lead.industry,
                "source": e.lead.source,
                "last_interaction_date": e.lead.last_interaction_date or "Unknown",
                "qualification_score": round(e.total_score, 1),
                "decision": e.decision,
                "priority": e.priority or "N/A",
                "primary_reason": e.primary_reason,
                "needs_review": "YES" if e.needs_review else "NO",
                "review_reasons": "; ".join(e.review_reasons) if e.review_reasons else "",
                "data_quality_issues": "; ".join(e.lead.data_quality_issues) if e.lead.data_quality_issues else "",
                "outreach_subject_variant_1": first_msg.subject if first_msg else "",
                "outreach_body_variant_1": first_msg.body.replace("\n", " ") if first_msg else ""
            })

    return output_path


def print_cli_summary(analytics: Dict[str, Any], evaluations: List[LeadEvaluation], json_path: str, csv_path: str):
    """Print an executive-level summary directly to terminal."""
    total = analytics.get("total_processed", 0)
    q_count = analytics.get("qualified_count", 0)
    q_pct = analytics.get("qualified_pct", 0)
    r_count = analytics.get("review_count", 0)
    r_pct = analytics.get("review_pct", 0)
    d_count = analytics.get("disqualified_count", 0)
    d_pct = analytics.get("disqualified_pct", 0)
    p1 = analytics.get("priority_distribution", {}).get("P1_HIGH", 0)
    p2 = analytics.get("priority_distribution", {}).get("P2_MEDIUM", 0)
    roi = analytics.get("business_roi", {})

    print("\n" + "=" * 70)
    print("           LEAD INTELLIGENCE SYSTEM - EXECUTIVE REPORT")
    print("=" * 70)
    print(f" Total Inbound Leads Ingested:    {total:,}")
    print(f" Processing Time:                 {roi.get('automated_processing_time_seconds', 0)}s ({roi.get('automated_processing_time_seconds', 0)/max(total, 1):.3f}s / lead)")
    engine_stats = Counter(e.processed_by for e in evaluations)
    engine_summary = ", ".join(f"{k}: {v}" for k, v in engine_stats.items())
    print(f" Evaluation Engine:               {engine_summary}")
    print("-" * 70)
    print(f" [✓] QUALIFIED (Ready for SDR):   {q_count:,} ({q_pct}%)")
    print(f"     - P1 High Priority (<= 2h):  {p1:,} leads")
    print(f"     - P2 Medium Priority (<= 24h):{p2:,} leads")
    print(f" [?] REVIEW (Human Sanity Check): {r_count:,} ({r_pct}%)")
    print(f" [✗] DISQUALIFIED (Filtered Out): {d_count:,} ({d_pct}%)")
    print("-" * 70)
    print(" BUSINESS IMPACT & SDR EFFICIENCY:")
    print(f" • Manual Baseline (10 min/lead): {roi.get('manual_qualification_hours_required', 0)} hours")
    print(f" • Estimated SDR Hours Saved:     {roi.get('sdr_hours_saved', 0)} hours ({roi.get('time_saved_percentage', 0)}% efficiency gain)")
    print(f" • Bottleneck Resolved:           Eliminates manual 8-12 min qualification drag")
    print("-" * 70)
    print(" COMMON DISQUALIFICATION REASONS:")
    for reason, count in analytics.get("common_disqualification_reasons", {}).items():
        print(f" • {reason}: {count} leads")
    print("-" * 70)
    print(" DELIVERABLES SAVED:")
    print(f" • Full JSON Report:  {json_path}")
    print(f" • CRM CSV Export:    {csv_path}")
    print("=" * 70 + "\n")
