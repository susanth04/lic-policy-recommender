# LIC Policy Recommender (v2 Package + v7 Dataset)

A transparent, brochure-verified LIC policy recommendation system combining natural language intent extraction, a strict catalogue eligibility gate, and a gradient-boosted decision ranker (GBR).

## System Architecture

```mermaid
flowchart TD
    Q[User Query Text + Optional Overrides] --> IP[1. Intent Parser & Scope Check]
    IP -->|Pension / Health / ULIP / Micro| OOS[Status: OUT_OF_SCOPE<br/>Plain explanation, 0 recs]
    IP -->|In-Scope Categories| EG[2. Hard Eligibility Gate<br/>catalogue.csv brochure rules]
    EG -->|Unknown / Missing Info| NI[Status: NEEDS_INFO<br/>Prompts for child_age, age, etc.]
    EG -->|0 Policies Pass| NEP[Status: NO_ELIGIBLE_POLICIES<br/>Explain limits exceeded]
    EG -->|Gate-Eligible Policies| GBR[3. GBR Ranker (gbr_only)<br/>Trained on verified plans]
    GBR -->|Top Relevance < 1.20| WM[Flag: weak_match = True<br/>Abstention Warning]
    GBR --> REP[4. Transparent Output<br/>Card, gate facts, brochure URL]
```

### Key Principles
1. **Hard Eligibility Gate First**: Policies that do not pass brochure entry age, policy term, sum assured, or gender rules are never recommended. Unknown limits remain not recommended.
2. **Exported Mode is `gbr_only`**: Experiments demonstrated that semantic (MiniLM) and cross-encoder scoring did not beat the baseline GBR with statistical significance (95% CI included 0). They remain optional diagnostic flags in `hybrid_config.json`, turned off by default.
3. **Knowledge Graph (KG) for Store & Explanations**: GraphML captures policy metadata and category linkages for generating structured explanation templates.
4. **Scope Limitation**: Only 21 brochure-verified plans (Endowment, Term Assurance, Money Back, Whole Life) are supported. Pension/annuity, health insurance, unit-linked (ULIP), and micro-insurance plans return `OUT_OF_SCOPE`.

---

## How to Run

### 1. Environment Setup
Install the lightweight GBR-only dependencies (requires no GPU, no PyTorch):
```bash
pip install -r model_out/requirements.txt
```

*(Optional: To run transformer diagnostic experiments, install `pip install -r model_out/requirements-semantic.txt`)*

### 2. Environment Variables
- `LIC_CORS_ORIGINS`: Comma-separated list of allowed origins (e.g., `http://localhost:3000,http://127.0.0.1:5500`), or `*` (default) for local testing.
- `PORT`: Port to bind the server (default: `8000`).

### 3. Start the API Server & Frontend
From the `model_out/` directory:
```bash
cd model_out
python -m uvicorn serve_fastapi:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at:
- **Interactive Web App**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **API Documentation (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Contract & Examples

### `GET /health`
Returns service operational state, plan counts, and abstention calibration parameters.
```json
{
  "ok": true,
  "status": "healthy",
  "mode": "gbr_only",
  "verified_plans": 21,
  "total_plans": 47,
  "trained_on": "2026-11-21",
  "data_version": "v7",
  "abstention_calibration": {
    "weak_match_threshold": 1.2,
    "auroc": 0.907,
    "match_recall": 1.0,
    "abstention_precision": 1.0,
    "abstention_recall": 0.286
  }
}
```

### `GET /policies`
Returns the catalogue of brochure-verified candidate plans.
Query parameters: `verified_only=true` (default: true).

---

### `POST /recommend`

#### Request Body Schema
```json
{
  "query": "I am 30 and want a pure term plan with 25 lakh cover for 20 years",
  "top_k": 5,
  "customer_age": 30,
  "child_age": null,
  "term": 20,
  "sum_assured": 2500000,
  "gender": "male"
}
```

#### Example 1: Status `OK` (Eligible Recommendations)
**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"query": "I am 30 and want a pure term plan with 25 lakh cover for 20 years"}'
```
**Response:**
```json
{
  "status": "OK",
  "query": "I am 30 and want a pure term plan with 25 lakh cover for 20 years",
  "eligible_count": 8,
  "weak_match": false,
  "recommendations": [
    {
      "rank": 1,
      "policy_id": "LIC_859",
      "policy_name": "LIC's Saral Jeevan Bima",
      "category": "Term Assurance",
      "predicted_relevance": 2.319,
      "kg_score": 0.65,
      "policy_url": "https://licindia.in/...",
      "brochure_url": "https://licindia.in/...",
      "explanation": {
        "gate_facts": [
          "eligibility gate result: eligible",
          "plan is active and in the recommendation pool",
          "age 30 is inside the catalogue entry-age range 18-65 (brochure-verified)",
          "term 20 years is inside the catalogue range 5-40 years (brochure-verified)",
          "sum assured Rs 2,500,000 is inside Rs 500,000-2,500,000 (brochure-verified)"
        ],
        "text": "Eligibility: ... Knowledge-graph links matched: goal 'Life Cover' ... Ranked by the gradient-boosted ranker trained on human grades."
      }
    }
  ],
  "understood_details": {
    "parsed_from_text": {"customer_age": 30, "term": 20, "sum_assured": 2500000},
    "structured_overrides": {},
    "effective_intent": {"customer_age": 30, "term": 20, "sum_assured": 2500000}
  }
}
```

#### Example 2: Status `OUT_OF_SCOPE`
**Request:**
```json
{"query": "I'm 58 and want a pension plan with guaranteed income"}
```
**Response:**
```json
{
  "status": "OUT_OF_SCOPE",
  "recommendations": [],
  "weak_match": true,
  "eligible_count": 0,
  "message": "The requested category (Pension / Annuity) is outside the scope of this recommender. This system only recommends brochure-verified LIC Endowment, Term Assurance, Money Back, and Whole Life plans. Pension/annuity, health insurance, ULIP (market-linked), and micro-insurance plans are not supported."
}
```

#### Example 3: Status `NEEDS_INFO` (Child Age Clarification)
**Request:**
```json
{"query": "parent saving for daughter education plan"}
```
**Response:**
```json
{
  "status": "NEEDS_INFO",
  "recommendations": [],
  "clarifications": [
    {
      "field": "child_age",
      "reason": "child age is required",
      "question": "How old is your child?",
      "policy_ids": ["LIC_732", "LIC_734"]
    }
  ]
}
```
*(When resubmitted with `{"child_age": 6}`, Jeevan Tarun and Children's Money Back qualify and are recommended).*

#### Example 4: Status `NO_ELIGIBLE_POLICIES`
**Request:**
```json
{"query": "I am 95 years old looking for a low premium policy with 5 lakh sum assured for 10 years"}
```
**Response:**
```json
{
  "status": "NO_ELIGIBLE_POLICIES",
  "eligible_count": 0,
  "recommendations": []
}
```

---

## Evaluation & Honesty Statement

1. **Model Performance**:
   - The GBR ranker achieves a nested leave-one-group-out NDCG@3 of **0.74** across 33 persona groups, measured against our own grading rubric.
   - This metric measures consistency with the project rubric and does not claim real-world recommendation quality.
2. **Dataset Transparency**:
   - The original dataset contained 2,204 labeled rows across 168 queries, of which 1,655 rows were initially drafted by an AI reviewer (`ai_provisional`) and 549 were human-reviewed.
   - The dataset in `data/` has had all AI markers and annotator tags removed (`query_id`, `group_id`, `policy_id`, `policy_name`, `grade`, `graded_at`, `notes`).
   - Accuracy claims are strictly omitted from the web UI.
3. **Abstention Calibration**:
   - Abstention AUROC is **0.907** on out-of-fold GBR predictions.
   - At threshold `WEAK_MATCH_BELOW = 1.20`: Match Recall is 100% (zero false negatives) and Abstention Precision is 100%.

---

## Known Limitations

1. **Jeevan Utsav (LIC_771) Term Limits**:
   - Jeevan Utsav is a Whole Life plan providing lifetime cover (up to age 100) with a premium-paying term (PPT) of 5–16 years.
   - The catalogue does not define fixed policy terms for it (`term_verified: false`). If a user query specifies a fixed policy term, Jeevan Utsav returns `unknown` under the eligibility gate and is not recommended.
2. **Scope Constraints**:
   - Covers only 21 brochure-verified products. Does not quote live premiums or perform medical underwriting.
3. **Disclaimer**:
   - This software is an academic decision-support prototype, not an official LIC portal, and does not provide financial or insurance advice. Always consult official LIC brochures before making financial decisions.

---

## Running the Test Suite

Run unit, contract, and regression probe tests:
```bash
python -m pytest tests/test_gate_unit.py tests/test_api_contracts.py tests/test_regression_probes.py -v
```


---

## Part A: Optional LLM Layer

### Setup

```bash
# Default: LLM fully disabled (no torch, no API key needed)
uvicorn app.main:app --port 8000

# Enable LLM layer (OpenAI example)
ENABLE_LLM=true LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini LLM_API_KEY=sk-... uvicorn app.main:app --port 8000
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENABLE_LLM` | `false` | Enable LLM layer (`true`/`false`) |
| `LLM_PROVIDER` | `fake` | `openai`, `gemini`, `fake` (for tests) |
| `LLM_MODEL` | `gpt-4o-mini` | Model name for the provider |
| `LLM_API_KEY` | _(none)_ | API key for the LLM provider |
| `LLM_TIMEOUT` | `3.0` | Hard total deadline in seconds (all retries included) |
| `LLM_MAX_TOKENS` | `512` | Max tokens per LLM completion |
| `LLM_DAILY_CAP` | `100` | Daily request cap; resets at midnight |

### LLM Safety Guarantees
1. **Never ranks or selects plans.** The gate (`eligibility_gate.py`) and GBR ranker run purely on brochure data.
2. **Never receives profile data.** Only the sanitised query text is sent to the LLM.
3. **Fact-constrained explanations.** A post-check rejects any generated text containing numbers, plan names, or claims not present in `gate_facts`; falls back to the deterministic template.
4. **Prompt-injection defence.** User text is stripped of control characters and wrapped in `<user_query>...</user_query>` delimiters; `</user_query>` within user text is HTML-escaped.
5. **Pydantic bounds validation.** LLM output is validated against strict field schemas before use; any failure causes immediate fallback to `parse_intent()`.
6. **Parser disagreement → user confirmation.** When rule and LLM parsers produce different values for the same field, neither is auto-applied; the user is asked to disambiguate.

### Paper Paragraph (LLM Evaluation)

> We integrated an optional, provider-agnostic LLM intent-extraction layer behind the existing rule parser in the LIC Policy Recommender. The LLM receives only sanitised query text (profile data is never transmitted) and must return a schema-validated JSON object; any bounds violation, unsupported key, low confidence (< 0.70), or hard timeout (3 s total including retries) triggers immediate fallback to the deterministic rule parser. We evaluated three conditions—rule-only, LLM-only, and hybrid (LLM with rule fallback)—on a human-annotated hold-out set of natural queries (`natural_queries_gold.csv`, dev/test split, no prompt tuning on test). Metrics include field-level exact-match accuracy (age, child age, term, sum assured, gender), downstream NDCG@3 and abstention AUROC, with group-level 1 000-fold bootstrap 95% confidence intervals. Newly eligible but unlabelled plans are explicitly assigned grade 0.0. We adopt the LLM configuration only if the downstream CI for ΔNDCG@3 or ΔAUROC excludes zero. If the numbers do not show improvement, we report this honestly and retain the rule parser as the production default.

---

## Part B: User Database & Profile Memory

### Setup

```bash
# Dev: SQLite (auto-created as lic_recommender.db)
uvicorn app.main:app --port 8000

# Prod: Postgres
DATABASE_URL=postgresql://user:pass@host/db uvicorn app.main:app --port 8000
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///lic_recommender.db` | SQLAlchemy database URL |
| `TOKEN_SECRET` | `change-me-before-deploy` | HMAC signing secret for device tokens |
| `TOKEN_EXPIRY_DAYS` | `365` | Device token validity in days |
| `HISTORY_RETENTION_DAYS` | `90` | Auto-delete query history after N days |

### API Contract

#### Auth
- `POST /auth/device` → `{user_id, device_token, notice}` — Anonymous session, no PII collected.

#### Consent (single source of truth)
- `POST /me/consent` → `{purpose: "profile_memory"|"query_history", granted: bool}` — Grant/revoke; revocation deletes associated data immediately.
- `DELETE /me/consent/{purpose}` — Shorthand revocation.

#### Profile
- `GET /me/profile` → `{profile, consent_status, allowed_keys}`
- `PUT /me/profile/{key}` — Upsert a confirmed attribute (`customer_age`, `child_age`, `gender`, `preferred_term`, `preferred_sum_assured` only).
- `DELETE /me/profile/{key}` — Remove one attribute.

#### History & Data
- `GET /me/history?limit=20` — Query history (only if `consent_history` granted).
- `DELETE /me/history/{id}` — Delete one item.
- `DELETE /me` — Hard delete everything (immediate, permanent).
- `GET /me/export` — Portable JSON export.

### Recommend with Memory
`POST /recommend` accepts an optional `X-Device-Token` header:
- If authenticated + `consent_profile=true`: confirmed profile fills unprovided fields (after form values and query-parsed values).
- Response includes `profile_applied` list and `needs_confirmation` list for stale age attributes.
- **Precedence**: explicit form fields > current query (LLM/rule parsed) > confirmed profile.
- OUT_OF_SCOPE responses are **never stored**, even with `consent_history=true`.

### Data Retention
Run the daily cleanup job (e.g. via Task Scheduler / cron):
```bash
py -3 tools/cleanup_history.py
py -3 tools/cleanup_history.py --dry-run  # preview only
```

### Data Minimisation & Privacy Notes
- Stored fields: `customer_age`, `child_age`, `gender`, `preferred_term`, `preferred_sum_assured` only.
- **Never stored**: health details, income, Aadhaar/PAN, phone, DOB, free-text beyond consented query history.
- Consent is opt-in; nothing is stored before explicit grant.
- This system is **designed to support DPDP-style data minimisation principles but has not been legally reviewed**. Consult a data protection officer before production deployment.

### Scheduled Retention Cleanup
```
# Windows Task Scheduler example (daily at 02:00)
schtasks /create /tn "LIC_Cleanup" /tr "py -3 C:\...\tools\cleanup_history.py" /sc daily /st 02:00
```

---

## Running the Full Test Suite

```bash
py -3 -m pytest tests/test_model_package_intent_parser.py tests/test_gate_unit.py tests/test_api_contracts.py tests/test_regression_probes.py tests/test_llm_safety.py tests/test_db_memory.py -v
# Expected: 45 passed
```

### Offline LLM Evaluation
```bash
py -3 tools/eval_llm_intent.py --split test   # test split (default)
py -3 tools/eval_llm_intent.py --split dev
py -3 tools/eval_llm_intent.py --split all
# Accepts custom gold file: --gold path/to/natural_queries_gold.csv
```
