"""
LLM Client and Batch Orchestrator.
Supports Gemini, OpenAI, Groq, and offline deterministic fallback.
Implements batch processing, exponential backoff, rate limit handling,
and robust JSON response parsing. Zero external dependencies.
"""

import json
import logging
import os
import re
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from src.models import Lead, LeadEvaluation, FactorScore, OutreachMessage
from src.rubric import QualificationRubric

def load_dotenv(env_path=".env"):
    """Load key-value pairs from .env file into os.environ if not already present."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


# Automatically load .env on module import
load_dotenv()
logger = logging.getLogger(__name__)




SYSTEM_PROMPT = """You are an expert Forward Deployed AI Engineer and Sales Qualification Intelligence engine for a mid-market B2B SaaS company.
Your objective is to evaluate a batch of inbound leads against our strict, defensible 4-factor qualification rubric, score each lead from 1.0 to 10.0, decide whether to QUALIFY, DISQUALIFY, or flag for REVIEW, assign an outreach priority, and craft high-converting personalized outreach messages.

### QUALIFICATION RUBRIC (Total Score: 1.0 - 10.0)
1. Company Size Fit (0.0 - 2.5 pts):
   - 100 - 2,500 FTE: 2.5 pts (Ideal ICP sweet spot)
   - 50 - 99 or 2,501 - 5,000 FTE: 2.0 pts (Strong scaling fit)
   - 20 - 49 or 5,001 - 10,000 FTE: 1.5 pts (Viable)
   - 10 - 19 FTE: 1.0 pt (Early stage)
   - > 10,000 FTE: 1.2 pts (Strategic Enterprise; flag for custom review)
   - < 10 FTE: 0.3 pts (Solopreneur/micro; below economic viability)
   - Missing/NA size: 1.0 pt (Flag for manual verification)

2. Industry Alignment (0.0 - 2.5 pts):
   - Tier 1 (SaaS, Software, Tech, FinTech, Finance, Analytics, Cybersecurity, AI/ML): 2.5 pts
   - Tier 2 (Healthcare, Logistics, Retail, Marketing, EdTech, Consulting, Insurance): 2.0 pts
   - Tier 3 (Manufacturing, Real Estate, Pharmaceuticals, Energy, Construction): 1.5 pts
   - Tier 4 (Hospitality, Events, Agriculture, Government, Local services): 0.8 pts
   - Unmapped/Missing: 1.0 pt (Flag for review)

3. Lead Source & Intent Velocity (0.0 - 3.0 pts):
   - Inbound Demo Request: 3.0 pts (Highest intent hand-raiser)
   - Referral / Sales Inbound Call: 2.5 pts (High trust)
   - Webinar Attendee: 2.0 pts (Active learner)
   - Content Download: 1.5 pts (Top-of-funnel research)
   - LinkedIn Outreach: 1.0 pt (Cold nurture)
   - Unknown/Other: 1.2 pts

4. Engagement Recency (0.0 - 2.0 pts):
   - <= 7 days: 2.0 pts (Hot lead)
   - 8 - 30 days: 1.5 pts (Active)
   - 31 - 90 days: 1.0 pt (Cooling)
   - 91 - 180 days: 0.5 pt (Cold)
   - > 180 days: 0.0 pts (Stale; triggers disqualification)
   - Missing date: 0.5 pt (Flag for review)

### DECISION RULES:
- QUALIFIED (Total >= 7.0): Fit verified. If >= 8.5, Priority is "P1_HIGH"; if 7.0 - 8.4, Priority is "P2_MEDIUM".
- DISQUALIFIED (Total < 5.0 OR Hard Disqualification):
  - Stale interaction (> 180 days).
  - Micro-org (< 5 FTE) with low-intent source.
  - Missing both contact and company name.
- REVIEW (Score 5.0 - 6.9 OR Critical Edge Case):
  - Borderline score.
  - Large enterprise (> 10,000 FTE).
  - Unmapped industry or missing company size.

### REQUIRED OUTPUT:
You MUST respond with ONLY a valid JSON object containing an "evaluations" array.
Format:
{
  "evaluations": [
    {
      "lead_id": <int>,
      "score": <float 1.0 - 10.0>,
      "decision": "QUALIFIED" | "DISQUALIFIED" | "REVIEW",
      "priority": "P1_HIGH" | "P2_MEDIUM" | null,
      "primary_reason": "<clear 1-2 sentence defense>",
      "factor_breakdown": {
        "company_size_fit": {"score": <float>, "reason": "<explanation>"},
        "industry_fit": {"score": <float>, "reason": "<explanation>"},
        "source_intent": {"score": <float>, "reason": "<explanation>"},
        "engagement_recency": {"score": <float>, "reason": "<explanation>"}
      },
      "needs_review": <boolean>,
      "review_reasons": ["<reason if needs_review is true>"],
      "outreach_messages": [
        {
          "variant": "Pain_Point_ROI",
          "subject": "<personalized subject>",
          "body": "<personalized body referencing company, name, vertical, and channel>"
        },
        {
          "variant": "Direct_Value_Prop",
          "subject": "<personalized subject>",
          "body": "<personalized body>"
        }
      ]
    }
  ]
}
"""


class LLMBatchClient:
    """Manages batch requests to LLM APIs with fallback to deterministic rubric."""

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        reference_date: Optional[str] = None,
        dry_run: bool = False
    ):
        self.dry_run = dry_run
        self.timeout = timeout
        self.max_retries = max_retries
        self.rubric = QualificationRubric(reference_date)

        # Check LLM_PROVIDER from environment if provider is 'auto' or not given
        env_provider = os.getenv("LLM_PROVIDER")
        if (not provider or provider.lower() == "auto") and env_provider:
            provider = env_provider.lower().strip()

        # Detect provider and credentials
        self.provider = provider.lower() if provider else "auto"
        if self.provider == "groq":
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
        elif self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        elif self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        elif self.provider in ("claude", "anthropic"):
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        else:
            self.api_key = (
                api_key
                or os.getenv("GROQ_API_KEY")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
            )

        if self.provider == "auto":
            if os.getenv("GROQ_API_KEY"):
                self.provider = "groq"
            elif os.getenv("GEMINI_API_KEY"):
                self.provider = "gemini"
            elif os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
            elif os.getenv("ANTHROPIC_API_KEY"):
                self.provider = "claude"
            elif self.api_key:
                self.provider = "groq"
            else:
                self.provider = "rule-based"

        # Set default model
        if model:
            self.model = model
        elif self.provider == "gemini":
            self.model = "gemini-1.5-flash"
        elif self.provider == "groq":
            self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        elif self.provider == "openai":
            self.model = "gpt-4o-mini"
        elif self.provider in ("claude", "anthropic"):
            self.model = "claude-3-haiku-20240307"
        else:
            self.model = "deterministic-rubric-v1"

    def set_reference_date(self, date_str: str):
        """Update anchor date for recency calculations."""
        self.rubric.set_reference_date(date_str)

    def process_batch(self, leads: List[Lead]) -> List[LeadEvaluation]:
        """
        Process a batch of leads.
        If dry_run, rule-based, or if API fails after retries,
        falls back to deterministic rubric so pipeline never breaks.
        """
        if self.dry_run or self.provider in ("rule-based", "offline", "mock") or not self.api_key:
            return [self.rubric.evaluate(lead) for lead in leads]

        # Attempt LLM batch evaluation
        prompt = self._build_batch_prompt(leads)
        raw_response = self._call_llm_with_retry(prompt)

        if not raw_response:
            logger.warning("LLM API returned empty response or failed all retries. Falling back to deterministic rubric.")
            return [self.rubric.evaluate(lead) for lead in leads]

        # Parse structured evaluations
        evaluations = self._parse_llm_response(raw_response, leads)
        if not evaluations or len(evaluations) != len(leads):
            logger.warning("LLM response parsing incomplete. Using deterministic fallback for missing evaluations.")
            return self._reconcile_with_rubric(evaluations, leads)

        return evaluations

    def _build_batch_prompt(self, leads: List[Lead]) -> str:
        """Create structured prompt containing the batch of leads."""
        leads_data = [lead.to_dict() for lead in leads]
        ref_date = self.rubric.reference_date_str or "2024-01-20"

        return (
            f"Reference Anchor Date: {ref_date}\n"
            f"Please evaluate the following batch of {len(leads)} leads:\n\n"
            f"{json.dumps(leads_data, indent=2)}\n\n"
            f"Respond ONLY with the specified JSON object containing the 'evaluations' array."
        )

    def _call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """Call LLM API with exponential backoff on 429 and 5xx errors."""
        backoff = 2.0
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.provider == "gemini":
                    return self._call_gemini_api(prompt)
                elif self.provider in ("openai", "groq"):
                    return self._call_openai_compatible_api(prompt)
                elif self.provider in ("claude", "anthropic"):
                    return self._call_claude_api(prompt)
                else:
                    return None
            except urllib.error.HTTPError as e:
                status = e.code
                err_msg = e.read().decode("utf-8", errors="replace")
                logger.warning(f"HTTP {status} on attempt {attempt}/{self.max_retries}: {err_msg[:200]}")
                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    retry_header = getattr(e, "headers", {}).get("retry-after") if hasattr(e, "headers") else None
                    if retry_header:
                        try:
                            sleep_time = float(retry_header) + 1.0
                        except ValueError:
                            sleep_time = backoff
                    else:
                        sleep_time = backoff
                    logger.info(f"Rate limited or server busy. Waiting {sleep_time:.1f}s before retry {attempt + 1}...")
                    time.sleep(sleep_time)
                    backoff *= 2
                else:
                    break
            except Exception as e:
                logger.warning(f"Network error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    break

        return None

    def _call_gemini_api(self, prompt: str) -> str:
        """Execute request to Google Gemini API using urllib."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "LeadIntelligenceSystem/1.0"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            # Extract text from response structure
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return ""

    def _call_openai_compatible_api(self, prompt: str) -> str:
        """Execute request to OpenAI or Groq endpoint using urllib."""
        if self.provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
        else:
            url = "https://api.openai.com/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.2
        }
        # Only add response_format if not using Groq gpt-oss which can trigger json_validate_failed on CoT
        if not (self.provider == "groq" and "gpt-oss" in self.model):
            payload["response_format"] = {"type": "json_object"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "LeadIntelligenceSystem/1.0"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""

    def _call_claude_api(self, prompt: str) -> str:
        """Execute request to Anthropic Claude API using urllib."""
        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": self.model if "claude" in self.model else "claude-3-haiku-20240307",
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.2
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            content = result.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", "")
            return ""

    def _parse_llm_response(self, text: str, leads: List[Lead]) -> List[LeadEvaluation]:
        """Safely parse LLM JSON output and construct LeadEvaluation objects."""
        # Clean markdown code fences if present
        clean_text = text.strip()
        if "```" in clean_text:
            match_fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
            if match_fence:
                clean_text = match_fence.group(1).strip()

        # If not matching fence, locate outer JSON braces
        if not (clean_text.startswith("{") or clean_text.startswith("[")):
            match_json = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", clean_text)
            if match_json:
                clean_text = match_json.group(0).strip()

        # Attempt robust parsing with fallback repair
        data = None
        try:
            data = json.loads(clean_text)
        except Exception:
            # Try removing trailing commas
            no_trailing = re.sub(r",\s*([\}\]])", r"\1", clean_text)
            try:
                data = json.loads(no_trailing)
            except Exception:
                # Try balancing truncated brackets
                for suffix in ("}", "]}", "]}}"):
                    try:
                        data = json.loads(no_trailing + suffix)
                        break
                    except Exception:
                        pass

        if not data:
            logger.warning(f"Failed to parse LLM JSON after repair attempts. Raw snippet: {clean_text[:200]}")
            return []

        if isinstance(data, list):
            eval_list = data
        elif isinstance(data, dict):
            eval_list = data.get("evaluations") or data.get("leads") or data.get("results") or []
        else:
            return []

        if not isinstance(eval_list, list):
            return []

        leads_by_id = {l.lead_id: l for l in leads}
        results = []

        for idx, item in enumerate(eval_list):
            if not isinstance(item, dict):
                continue
            raw_id = item.get("lead_id") or item.get("id") or item.get("index")
            lead = None
            try:
                num_id = int(raw_id)
                if num_id in leads_by_id:
                    lead = leads_by_id[num_id]
            except (ValueError, TypeError):
                pass

            if lead is None and idx < len(leads):
                lead = leads[idx]

            if lead is None:
                continue
            lead_id = lead.lead_id

            # Parse factor breakdown
            factors = {}
            for fname, fdata in item.get("factor_breakdown", {}).items():
                if isinstance(fdata, dict):
                    factors[fname] = FactorScore(
                        factor_name=fname,
                        score=float(fdata.get("score", 0.0)),
                        max_score=2.5 if "recency" not in fname else 2.0,
                        reason=fdata.get("reason", "")
                    )

            # Parse outreach messages
            messages = []
            for m in item.get("outreach_messages", []):
                if isinstance(m, dict):
                    messages.append(OutreachMessage(
                        variant=m.get("variant", "Standard"),
                        subject=m.get("subject", ""),
                        body=m.get("body", "")
                    ))

            # Normalize decision
            decision = str(item.get("decision", "REVIEW")).upper()
            if decision in ("REJECTED", "REJECT", "DISQUALIFY"):
                decision = "DISQUALIFIED"

            score = float(item.get("score") or item.get("qualification_score") or 5.0)
            priority = item.get("priority") or item.get("priority_rank")

            if decision == "QUALIFIED" and not messages:
                messages = self.rubric.generate_outreach_messages(lead, score, priority)

            results.append(LeadEvaluation(
                lead_id=lead_id,
                lead=lead,
                total_score=score,
                decision=decision,
                priority=priority,
                factor_breakdown=factors or self.rubric.evaluate(lead).factor_breakdown,
                primary_reason=item.get("primary_reason") or item.get("reasoning") or "LLM batch evaluation",
                outreach_messages=messages,
                needs_review=bool(item.get("needs_review", False)),
                review_reasons=item.get("review_reasons", []),
                processed_by=f"llm_{self.provider}"
            ))

        return results

    def _reconcile_with_rubric(
        self, evaluations: List[LeadEvaluation], leads: List[Lead]
    ) -> List[LeadEvaluation]:
        """Fill in any leads missing from LLM output using deterministic rubric."""
        evaluated_ids = {e.lead_id for e in evaluations}
        reconciled = list(evaluations)

        for lead in leads:
            if lead.lead_id not in evaluated_ids:
                reconciled.append(self.rubric.evaluate(lead))

        # Maintain original order
        lead_order = {l.lead_id: i for i, l in enumerate(leads)}
        reconciled.sort(key=lambda e: lead_order.get(e.lead_id, 0))
        return reconciled
