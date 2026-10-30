# Lead Intelligence System (FDE Edition)

> **Forward Deployed AI Engineering Solution** for automated B2B lead qualification, priority routing, and personalized outreach generation.

---

## 1. Problem Statement & Business Context
A mid-market B2B SaaS company receives **~1,200 inbound leads/month**, but sales only qualifies and contacts **~60 (a 95% loss rate)**. Sales reps spend **8–12 minutes manually researching each lead**, creating a massive qualification drag and letting high-intent pipeline decay. 

This system automates intake, applies an explainable 4-factor scoring rubric, segments leads into actionable priority queues, and generates tailored first-touch outreach in **under 0.05 seconds per lead**, recovering **160+ hours/month of SDR selling capacity**.

---

## 2. Qualification Rubric

The system scores each lead on a **1.0 to 10.0 scale** across 4 explicit business dimensions:
1. **Company Size / ICP Fit (25% | Max 2.5 pts):** Sweet spot is **100–2,500 FTE (2.5 pts)** and **50–99 / 2,501–5,000 FTE (2.0 pts)**. Startups (<10 FTE) receive 0.3 pts (lack budget); enterprises (>10,000 FTE) receive 1.2 pts and trigger Strategic Enterprise routing.
2. **Industry Alignment (25% | Max 2.5 pts):** **Tier-1 core verticals (2.5 pts)** (SaaS, FinTech, Cybersecurity, Analytics, Cloud) have direct product fit. **Tier-2 growth sectors (2.0 pts)** (Healthcare, Logistics, Retail, Marketing) represent high-volume use cases. Unmapped industries receive 1.0 pt and are flagged for review.
3. **Source Intent Velocity (30% | Max 3.0 pts):** Inbound demo requests (**3.0 pts**) and direct referrals/sales calls (**2.5 pts**) reflect active buyer interest. Webinars (**2.0 pts**), content downloads (**1.5 pts**), and cold LinkedIn outreach (**1.0 pt**) reflect progressive awareness stages.
4. **Engagement Recency (20% | Max 2.0 pts):** Evaluated against reference anchor (≤7 days = **2.0 pts**; 8–30 days = **1.5 pts**; 31–90 days = **1.0 pt**; 91–180 days = **0.5 pt**; >180 days = **0.0 pts** & triggers automatic disqualification).

### Decision Thresholds:
- **`QUALIFIED` (Score ≥ 7.0):** Immediate SDR action.
  - **`P1_HIGH` (Score ≥ 8.5):** Outreach SLA ≤ 2 hours.
  - **`P2_MEDIUM` (Score 7.0–8.4):** Outreach SLA ≤ 24 hours.
- **`REVIEW` (Score 5.0–6.9 OR Edge Cases):** Borderline leads, unmapped industries, or missing headcounts flagged for a 60-second SDR sanity check.
- **`DISQUALIFIED` (Score < 5.0 OR Hard Disqualification):** Leads with decayed intent (>180 days stale), micro-headcount from cold channels (<5 FTE), or missing both company and contact names.

---

## 3. Quickstart & Usage

The codebase is built with **zero external dependencies** (pure Python Standard Library). It runs on any standard Python 3.8+ environment across Linux, macOS, and Windows.

```bash
# 1. Run full-scale production benchmark (100 leads)
python main.py -i docs/leads_full_scale_100.csv -o output_report.json --csv-output output_report.csv

# 2. Run edge-case / robustness test suite (20 messy leads)
python main.py -i docs/leads_testing.csv -o output_testing.json --csv-output output_testing.csv

# 3. Ingest default 30-lead dataset
python main.py

# 4. Run with an LLM API (Gemini, OpenAI, or Groq) in batches of 5
python main.py -i docs/leads_full_scale_100.csv --provider gemini --api-key $GEMINI_API_KEY --batch-size 5

# 5. Run prompt & rubric validation test (5-10 diverse leads)
python test_prompts.py --dry-run

# 6. Run automated unit test suite
python -m unittest discover -s tests -p "test_*.py"
```

---

## 4. Deliverables & Output Formats

1. **`output_report.json`:** Structured JSON containing:
   - `executive_summary`: Processed counts, qualification %, SDR hours saved, industry/size distributions.
   - `priority_queue_qualified`: Qualified leads sorted descending by score with full factor breakdowns and 3 personalized message variants (*Pain Point ROI*, *Direct Value Prop*, *Executive Briefing*).
   - `review_queue_flagged`: Ambiguous cases with specific human review triggers.
   - `disqualified_archive`: Disqualified leads with primary root-cause rejection reasons.
   - `sample_outreach_messages`: 5 featured multi-variant message samples ready for SDR copy-paste.
2. **`output_report.csv`:** Flat CSV table formatted for instant import into Salesforce or HubSpot CRM.

---

## 5. Robustness & Edge Case Handling

- **Batch Processing:** Processes leads in chunks (default 5–10 leads per batch) rather than inefficient 1-by-1 API calls.
- **Graceful API Degradation:** Implements exponential backoff on HTTP 429 / 5xx errors. If an LLM endpoint fails or lacks an API key, the system transparently falls back to the deterministic rubric engine, ensuring **zero crashes**.
- **Malformed & Dirty Data:**
  - `leads_testing.csv` validated with missing contact names, blank companies, `NA` company sizes, corrupt dates, unmapped industries, and 50,000+ FTE mega-enterprises.
  - Automatically sanitizes missing fields, assigns conservative baseline scores, and flags items to `needs_review`.
- **Dynamic Date Anchoring:** Automatically detects maximum date in historical datasets (`2024-01-20`) so historical records are not mistakenly flagged as stale.

---

## 6. Known Limitations
1. **Public vs. Private Firm Headcounts:** For stealth startups or private firms with `NA` headcount, the system relies on a neutral baseline and flags for manual review; integration with Clearbit or Apollo API would automate headcount enrichment.
2. **Dynamic Buying Window:** Intent is currently evaluated from interaction dates and channels; incorporating web page visit frequency or product usage telemetry would further sharpen qualification accuracy.
