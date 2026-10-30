"""
Data models and schemas for the Lead Intelligence System.
Pure Python standard library implementation for zero-dependency portability.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class Lead:
    """Represents an ingested lead record."""
    lead_id: int
    name: str
    company: str
    company_size: Optional[int]
    industry: str
    source: str
    last_interaction_date: Optional[str]
    raw_data: Dict[str, Any] = field(default_factory=dict)
    data_quality_issues: List[str] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: Dict[str, str], lead_id: int) -> "Lead":
        """Parse and sanitize a raw CSV row into a Lead model."""
        issues = []
        name = row.get("name", "").strip()
        company = row.get("company", "").strip()
        industry = row.get("industry", "").strip()
        source = row.get("source", "").strip()
        raw_size = row.get("company_size", "").strip()
        raw_date = row.get("last_interaction_date", "").strip()

        # Sanitize name
        if not name or name.upper() in ("NA", "N/A", "UNKNOWN", "NONE"):
            name = "Valued Leader"
            issues.append("missing_contact_name")

        # Sanitize company
        if not company or company.upper() in ("NA", "N/A", "UNKNOWN", "NONE"):
            company = "Your Organization"
            issues.append("missing_company_name")

        # Sanitize company_size
        company_size: Optional[int] = None
        if raw_size and raw_size.upper() not in ("NA", "N/A", "UNKNOWN", "NONE", ""):
            try:
                # Remove any commas, plus signs, e.g. "8,000" or "5000+"
                clean_size = raw_size.replace(",", "").replace("+", "")
                company_size = int(float(clean_size))
                if company_size <= 0:
                    company_size = None
                    issues.append("invalid_company_size_zero_or_negative")
            except (ValueError, TypeError):
                company_size = None
                issues.append("unparseable_company_size")
        else:
            issues.append("missing_company_size")

        # Sanitize industry
        if not industry or industry.upper() in ("NA", "N/A", "UNKNOWN", "NONE", "UNKNOWN SECTOR"):
            industry = "Unknown"
            issues.append("unmapped_industry")

        # Sanitize source
        if not source or source.upper() in ("NA", "N/A", "UNKNOWN", "NONE", "UNKNOWN SOURCE"):
            source = "Direct Inbound"
            issues.append("missing_or_unknown_source")

        # Sanitize date
        valid_date = None
        if raw_date and raw_date.upper() not in ("NA", "N/A", "UNKNOWN", "NONE", ""):
            try:
                # Validate YYYY-MM-DD
                datetime.strptime(raw_date, "%Y-%m-%d")
                valid_date = raw_date
            except ValueError:
                issues.append("invalid_date_format")
        else:
            issues.append("missing_last_interaction_date")

        return cls(
            lead_id=lead_id,
            name=name,
            company=company,
            company_size=company_size,
            industry=industry,
            source=source,
            last_interaction_date=valid_date,
            raw_data=dict(row),
            data_quality_issues=issues
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lead_id": self.lead_id,
            "name": self.name,
            "company": self.company,
            "company_size": self.company_size if self.company_size is not None else "Unknown",
            "industry": self.industry,
            "source": self.source,
            "last_interaction_date": self.last_interaction_date or "Unknown",
            "data_quality_issues": self.data_quality_issues
        }


@dataclass
class FactorScore:
    """Individual rubric factor evaluation."""
    factor_name: str
    score: float
    max_score: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutreachMessage:
    """Personalized outreach copy."""
    variant: str
    subject: str
    body: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LeadEvaluation:
    """Complete evaluation result for a lead."""
    lead_id: int
    lead: Lead
    total_score: float  # Scale 1.0 - 10.0
    decision: str  # QUALIFIED, DISQUALIFIED, REVIEW
    priority: Optional[str]  # P1_HIGH, P2_MEDIUM, or None
    factor_breakdown: Dict[str, FactorScore]
    primary_reason: str
    outreach_messages: List[OutreachMessage] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    processed_by: str = "deterministic_rubric"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lead_id": self.lead_id,
            "lead": self.lead.to_dict(),
            "score": round(self.total_score, 1),
            "qualification_score": round(self.total_score, 1),
            "decision": self.decision,
            "priority": self.priority,
            "priority_rank": self.priority,
            "reasoning": self.primary_reason,
            "primary_reason": self.primary_reason,
            "factor_breakdown": {
                k: v.to_dict() for k, v in self.factor_breakdown.items()
            },
            "outreach_messages": [m.to_dict() for m in self.outreach_messages],
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "processed_by": self.processed_by
        }
