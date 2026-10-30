"""
Qualification rubric engine and outreach generator.
Provides deterministic scoring, factor evaluations, decision routing,
and personalized multi-variant messaging.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from src.models import Lead, FactorScore, OutreachMessage, LeadEvaluation


# Industry classification tiers
TIER_1_INDUSTRIES = {
    "saas", "software", "technology", "fintech", "finance",
    "analytics", "cybersecurity", "ai", "cloud", "data"
}
TIER_2_INDUSTRIES = {
    "healthcare", "logistics", "retail", "marketing", "edtech",
    "consulting", "insurance", "e-commerce", "telecom", "media",
    "real estate platform", "digital marketing"
}
TIER_3_INDUSTRIES = {
    "manufacturing", "real estate", "pharmaceuticals", "energy",
    "construction", "biotechnology", "aerospace", "automotive"
}
TIER_4_INDUSTRIES = {
    "hospitality", "events", "agriculture", "government", "restaurant",
    "local services", "non-profit", "entertainment"
}


class QualificationRubric:
    """Encapsulates the 4-factor scoring rubric and decision logic."""

    def __init__(self, reference_date: Optional[str] = None):
        """
        reference_date: YYYY-MM-DD anchor date for recency calculation.
        If None, will be dynamically inferred or default to today.
        """
        self.reference_date_str = reference_date
        self.reference_date = (
            datetime.strptime(reference_date, "%Y-%m-%d")
            if reference_date else None
        )

    def set_reference_date(self, date_str: str):
        """Set or update reference date."""
        self.reference_date_str = date_str
        self.reference_date = datetime.strptime(date_str, "%Y-%m-%d")

    def _eval_company_size(self, lead: Lead) -> FactorScore:
        """Evaluate company size against mid-market ICP (Max: 2.5 pts)."""
        size = lead.company_size

        if size is None:
            return FactorScore(
                factor_name="company_size_fit",
                score=1.0,
                max_score=2.5,
                reason="Company size unknown/missing; assigned neutral baseline (+1.0). Requires verification."
            )

        if 100 <= size <= 2500:
            return FactorScore(
                factor_name="company_size_fit",
                score=2.5,
                max_score=2.5,
                reason=f"Company size ({size:,}) matches ideal mid-market ICP sweet spot (100-2,500 FTE)."
            )
        elif 50 <= size < 100 or 2501 <= size <= 5000:
            return FactorScore(
                factor_name="company_size_fit",
                score=2.0,
                max_score=2.5,
                reason=f"Company size ({size:,}) is a strong scaling fit (50-99 or 2,501-5,000 FTE)."
            )
        elif 20 <= size < 50 or 5001 <= size <= 10000:
            return FactorScore(
                factor_name="company_size_fit",
                score=1.5,
                max_score=2.5,
                reason=f"Company size ({size:,}) is viable (20-49 FTE or 5,000-10,000 FTE)."
            )
        elif 10 <= size < 20:
            return FactorScore(
                factor_name="company_size_fit",
                score=1.0,
                max_score=2.5,
                reason=f"Early stage startup size ({size:,} FTE); budget may be constrained."
            )
        elif size > 10000:
            return FactorScore(
                factor_name="company_size_fit",
                score=1.2,
                max_score=2.5,
                reason=f"Mega enterprise ({size:,} FTE); high potential ACV but complex procurement cycles."
            )
        else:  # size < 10 (solopreneur / micro-business)
            return FactorScore(
                factor_name="company_size_fit",
                score=0.3,
                max_score=2.5,
                reason=f"Micro business/solopreneur ({size:,} FTE); below minimum economic viability."
            )

    def _eval_industry(self, lead: Lead) -> FactorScore:
        """Evaluate industry vertical fit (Max: 2.5 pts)."""
        ind_clean = lead.industry.lower().strip()

        # Check Tier 1
        for key in TIER_1_INDUSTRIES:
            if key in ind_clean:
                return FactorScore(
                    factor_name="industry_fit",
                    score=2.5,
                    max_score=2.5,
                    reason=f"'{lead.industry}' is a Tier-1 core vertical with high SaaS adoption and tech maturity."
                )

        # Check Tier 2
        for key in TIER_2_INDUSTRIES:
            if key in ind_clean:
                return FactorScore(
                    factor_name="industry_fit",
                    score=2.0,
                    max_score=2.5,
                    reason=f"'{lead.industry}' is a Tier-2 growth sector with established workflow automation needs."
                )

        # Check Tier 3
        for key in TIER_3_INDUSTRIES:
            if key in ind_clean:
                return FactorScore(
                    factor_name="industry_fit",
                    score=1.5,
                    max_score=2.5,
                    reason=f"'{lead.industry}' is a secondary sector with longer sales cycle."
                )

        # Check Tier 4
        for key in TIER_4_INDUSTRIES:
            if key in ind_clean:
                return FactorScore(
                    factor_name="industry_fit",
                    score=0.8,
                    max_score=2.5,
                    reason=f"'{lead.industry}' has historically low conversion / non-target alignment."
                )

        if ind_clean in ("unknown", "unknown sector", ""):
            return FactorScore(
                factor_name="industry_fit",
                score=1.0,
                max_score=2.5,
                reason="Industry unmapped/missing; assigned neutral baseline (+1.0). Flagged for review."
            )

        # General unclassified business
        return FactorScore(
            factor_name="industry_fit",
            score=1.5,
            max_score=2.5,
            reason=f"'{lead.industry}' is an adjacent vertical with moderate software fit."
        )

    def _eval_source(self, lead: Lead) -> FactorScore:
        """Evaluate lead source and inbound intent velocity (Max: 3.0 pts)."""
        src_clean = lead.source.lower().strip()

        if "demo" in src_clean or "inbound demo" in src_clean:
            return FactorScore(
                factor_name="source_intent",
                score=3.0,
                max_score=3.0,
                reason="Inbound demo request represents active, high-intent buyer hand-raising."
            )
        elif "referral" in src_clean:
            return FactorScore(
                factor_name="source_intent",
                score=2.5,
                max_score=3.0,
                reason="Referral channel carries high trust and strong conversion probability."
            )
        elif "sales call" in src_clean or "inbound call" in src_clean:
            return FactorScore(
                factor_name="source_intent",
                score=2.5,
                max_score=3.0,
                reason="Direct sales call indicates bilateral conversation and engaged interest."
            )
        elif "webinar" in src_clean:
            return FactorScore(
                factor_name="source_intent",
                score=2.0,
                max_score=3.0,
                reason="Webinar attendee demonstrates active interest in problem domain."
            )
        elif "content" in src_clean or "download" in src_clean:
            return FactorScore(
                factor_name="source_intent",
                score=1.5,
                max_score=3.0,
                reason="Content download reflects top-of-funnel research; requires qualification."
            )
        elif "linkedin" in src_clean or "outreach" in src_clean:
            return FactorScore(
                factor_name="source_intent",
                score=1.0,
                max_score=3.0,
                reason="Outbound LinkedIn touchpoint; unproven intent, requires active nurturing."
            )
        else:
            return FactorScore(
                factor_name="source_intent",
                score=1.2,
                max_score=3.0,
                reason=f"Channel '{lead.source}' has unspecified intent; assigned baseline (+1.2)."
            )

    def _eval_recency(self, lead: Lead) -> Tuple[FactorScore, int]:
        """Evaluate recency of interaction against reference anchor (Max: 2.0 pts)."""
        if not lead.last_interaction_date:
            return FactorScore(
                factor_name="engagement_recency",
                score=0.5,
                max_score=2.0,
                reason="Missing interaction date; assigned conservative recency score (+0.5)."
            ), 999

        try:
            lead_dt = datetime.strptime(lead.last_interaction_date, "%Y-%m-%d")
        except ValueError:
            return FactorScore(
                factor_name="engagement_recency",
                score=0.5,
                max_score=2.0,
                reason="Invalid date format; assigned conservative recency score (+0.5)."
            ), 999

        ref_dt = self.reference_date or datetime.now()
        days_diff = (ref_dt - lead_dt).days

        if days_diff < 0:
            # Future date or clock skew, treat as today
            days_diff = 0

        if days_diff <= 7:
            return FactorScore(
                factor_name="engagement_recency",
                score=2.0,
                max_score=2.0,
                reason=f"Touched {days_diff}d ago (<= 7 days): Hot lead in prime conversion window."
            ), days_diff
        elif days_diff <= 30:
            return FactorScore(
                factor_name="engagement_recency",
                score=1.5,
                max_score=2.0,
                reason=f"Touched {days_diff}d ago (8-30 days): Active opportunity, prompt follow-up needed."
            ), days_diff
        elif days_diff <= 90:
            return FactorScore(
                factor_name="engagement_recency",
                score=1.0,
                max_score=2.0,
                reason=f"Touched {days_diff}d ago (31-90 days): Cooling lead, needs re-engagement hook."
            ), days_diff
        elif days_diff <= 180:
            return FactorScore(
                factor_name="engagement_recency",
                score=0.5,
                max_score=2.0,
                reason=f"Touched {days_diff}d ago (91-180 days): Cold lead with significant intent decay."
            ), days_diff
        else:
            return FactorScore(
                factor_name="engagement_recency",
                score=0.0,
                max_score=2.0,
                reason=f"Touched {days_diff}d ago (>180 days): Stale lead exceeding viable sales window."
            ), days_diff

    def evaluate(self, lead: Lead) -> LeadEvaluation:
        """Run full rubric evaluation and generate decision, priority, and outreach."""
        # 1. Evaluate individual factors
        f_size = self._eval_company_size(lead)
        f_ind = self._eval_industry(lead)
        f_src = self._eval_source(lead)
        f_rec, days_diff = self._eval_recency(lead)

        factors = {
            "company_size_fit": f_size,
            "industry_fit": f_ind,
            "source_intent": f_src,
            "engagement_recency": f_rec
        }

        raw_total = f_size.score + f_ind.score + f_src.score + f_rec.score
        total_score = max(1.0, min(10.0, raw_total))

        review_reasons = []
        needs_review = False
        decision = "QUALIFIED"
        priority = None
        primary_reason = ""

        # Check hard disqualification conditions
        if days_diff > 180:
            decision = "DISQUALIFIED"
            primary_reason = f"Stale interaction ({days_diff} days ago). Lead intent has decayed beyond acceptable window."
        elif lead.company_size is not None and lead.company_size < 5 and ("linkedin" in lead.source.lower() or "content" in lead.source.lower()):
            decision = "DISQUALIFIED"
            primary_reason = f"Micro-organization ({lead.company_size} FTE) acquired via low-intent channel ({lead.source}). Below ICP threshold."
        elif "missing_company_name" in lead.data_quality_issues and "missing_contact_name" in lead.data_quality_issues:
            decision = "DISQUALIFIED"
            primary_reason = "Missing critical identity fields (both company name and contact name missing)."
        elif total_score < 5.0:
            decision = "DISQUALIFIED"
            primary_reason = f"Overall qualification score ({total_score:.1f}/10) below minimum threshold of 5.0."

        # Check review triggers (Human-in-the-loop)
        if decision != "DISQUALIFIED":
            if lead.company_size is not None and lead.company_size > 10000:
                needs_review = True
                review_reasons.append(f"Strategic Enterprise ({lead.company_size:,} FTE): Route to Strategic Enterprise AE team.")
            if "unmapped_industry" in lead.data_quality_issues:
                needs_review = True
                review_reasons.append("Unmapped industry: SDR to verify vertical alignment.")
            if "missing_company_size" in lead.data_quality_issues:
                needs_review = True
                review_reasons.append("Missing company size: Verify headcount via LinkedIn/Clearbit before demo.")
            if 5.0 <= total_score < 7.0:
                decision = "REVIEW"
                needs_review = True
                review_reasons.append(f"Borderline score ({total_score:.1f}/10): Balance of positive signals and friction factors.")
                primary_reason = f"Moderate fit ({total_score:.1f}/10) flagged for quick 60-second SDR review."

        # Finalize QUALIFIED leads
        if decision == "QUALIFIED":
            if total_score >= 8.5:
                priority = "P1_HIGH"
                primary_reason = f"High-priority fit ({total_score:.1f}/10): Core ICP vertical ({lead.industry}), strong size ({lead.company_size or 'target'}), and high-intent source."
            else:
                priority = "P2_MEDIUM"
                primary_reason = f"Solid qualified lead ({total_score:.1f}/10): Good alignment with mid-market parameters."

        # Generate outreach messages for QUALIFIED leads (and sample for REVIEW)
        outreach_messages = []
        if decision == "QUALIFIED" or (decision == "REVIEW" and total_score >= 6.0):
            outreach_messages = self.generate_outreach_messages(lead, total_score, priority)

        return LeadEvaluation(
            lead_id=lead.lead_id,
            lead=lead,
            total_score=total_score,
            decision=decision,
            priority=priority,
            factor_breakdown=factors,
            primary_reason=primary_reason,
            outreach_messages=outreach_messages,
            needs_review=needs_review,
            review_reasons=review_reasons,
            processed_by="deterministic_rubric"
        )

    def generate_outreach_messages(
        self, lead: Lead, score: float, priority: Optional[str]
    ) -> List[OutreachMessage]:
        """Generate 3 personalized, on-brand outreach variants for a qualified lead."""
        name = lead.name.split()[0] if lead.name else "there"
        company = lead.company if lead.company != "Your Organization" else "your team"
        industry = lead.industry if lead.industry != "Unknown" else "B2B SaaS"
        source = lead.source.lower()

        # Channel hook
        if "demo" in source:
            hook = f"Following up on your recent demo request for {company}"
            context_snippet = "noticed your interest in accelerating your qualification pipeline"
        elif "webinar" in source:
            hook = f"Glad you could attend our recent session on revenue acceleration"
            context_snippet = f"saw you joined our webinar on modernizing B2B sales workflows"
        elif "referral" in source:
            hook = f"Reaching out following an executive referral regarding {company}"
            context_snippet = f"our mutual partners highlighted {company}'s impressive growth in {industry}"
        elif "call" in source:
            hook = f"Continuing our discussion regarding {company}'s sales operations"
            context_snippet = f"building on our recent conversation around pipeline velocity"
        elif "content" in source:
            hook = f"Saw you recently reviewed our guide on B2B lead acceleration"
            context_snippet = f"wanted to share how other {industry} leaders are putting those insights into practice"
        else:
            hook = f"Connecting with {company} regarding sales automation"
            context_snippet = f"impressed by {company}'s momentum in the {industry} space"

        # Variant 1: Pain Point & ROI
        v1_subject = f"{company} <> Lead Intelligence: Cutting 95% qualification drop-off"
        v1_body = (
            f"Hi {name},\n\n"
            f"{hook}. Many scaling {industry} teams tell us their SDRs spend 8-12 minutes "
            f"per inbound lead manually researching fit—resulting in slow response times and missed opportunities.\n\n"
            f"We built an automated Lead Intelligence engine that pre-qualifies inbound prospects in seconds, "
            f"surfaces high-intent buyers, and drafts personalized outreach automatically.\n\n"
            f"Would you be open to a 10-minute briefing next Tuesday to see how {company} could recover 15+ hours "
            f"per SDR weekly?\n\n"
            f"Best regards,\n"
            f"Antigravity Sales Team"
        )

        # Variant 2: Direct Value Prop
        v2_subject = f"Accelerating {company}'s inbound pipeline"
        v2_body = (
            f"Hi {name},\n\n"
            f"I {context_snippet}. Given {company}'s current trajectory, ensuring every high-value "
            f"prospect receives immediate, personalized engagement is critical.\n\n"
            f"Our platform helped similar {industry} mid-market teams increase their lead-to-opportunity "
            f"conversion by 3.2x while cutting manual qualification overhead to zero.\n\n"
            f"Do you have 15 minutes this Thursday afternoon for a quick look at the workflow?\n\n"
            f"Best,\n"
            f"Antigravity Growth Team"
        )

        # Variant 3: Executive Briefing
        v3_subject = f"Strategic efficiency for {company}'s revenue org"
        v3_body = (
            f"Hi {name},\n\n"
            f"Reaching out directly as {company} expands its footprint in {industry}. "
            f"Mid-market revenue leaders are under pressure to do more with leaner teams without sacrificing pipeline velocity.\n\n"
            f"We've engineered a continuous qualification framework that sequences outreach based on live intent "
            f"and account fit, eliminating the 95% drop-off typical of manual SDR workflows.\n\n"
            f"I've put together a 1-page executive brief tailored for {company}. Would you prefer I send that over "
            f"or would a brief 10-minute chat be easier?\n\n"
            f"Warmly,\n"
            f"Antigravity Enterprise Solutions"
        )

        return [
            OutreachMessage(variant="Pain_Point_ROI", subject=v1_subject, body=v1_body),
            OutreachMessage(variant="Direct_Value_Prop", subject=v2_subject, body=v2_body),
            OutreachMessage(variant="Executive_Briefing", subject=v3_subject, body=v3_body)
        ]
