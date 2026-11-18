# LIC Policy Recommender: Technical Documentation

**Repository snapshot:** 19 November 2026  
**System status:** reproducible notebook smoke-test pipeline; not a production recommender or validated financial-advice system.

## 1. Purpose and scope

This project explores LIC policy discovery using a policy catalogue, rule-based eligibility checks, intent extraction, semantic and lexical retrieval, a small knowledge-graph preference layer, and a cross-encoder reranker. The current pipeline is designed to verify data flow and software behavior with synthetic queries and labels. It does not establish real-world recommendation quality.

The canonical pipeline is notebooks `01`–`04`. `config.json` points notebook 01 at `LIC_Policy_Recommendation_System_New_v3.ipynb` to read its synthetic persona/query templates. Older/reference notebooks are kept under `archive/legacy_notebooks/`; they are not stages run by `tools/run_notebooks.py`.

## 2. System architecture

```text
data/policies.csv + data/source/eligibility_from_brochures.csv
                        |
                        v
01_data_and_catalogue.ipynb
  - validate and merge the catalogue
  - extract synthetic queries
  - remap IDs and prepare smoke labels
  - write grouped and policy-held-out splits
                        |
          +-------------+------------------+
          v                                v
02_intent_and_kg.ipynb             03_finetune_biencoder.ipynb
  - parse intent                      - train synthetic MNRL triplets
  - apply eligibility rules            - cache checkpoints
  - export GraphML and intents                  |
          +-------------------------------------+
                        v
04_retrieval_fusion_rerank.ipynb
  - filter to eligible policies
  - combine dense, BM25, KG and cross-encoder signals
  - tune on validation profiles
  - write smoke-test reports
```

Notebooks exchange files through `artifacts/` and `results/`, rather than depending on another notebook's in-memory variables. The runner executes them in order:

```powershell
python tools\run_notebooks.py
python -m pytest -q
```

Alternatively, use `make reproduce` and `make test` where `make` is installed.

## 3. Repository map

| Path | Role |
|---|---|
| `01_data_and_catalogue.ipynb` | Catalogue validation/merge, synthetic query extraction, label handling, and split generation. |
| `02_intent_and_kg.ipynb` | Regex-based intent parser, shared gate integration, recommendation smoke function, and GraphML export. |
| `03_finetune_biencoder.ipynb` | Synthetic-only bi-encoder fine-tuning with Multiple Negatives Ranking Loss (MNRL). |
| `04_retrieval_fusion_rerank.ipynb` | Retrieval, score fusion, cross-encoder reranking, validation tuning, and test-profile smoke output. |
| `LIC_Policy_Recommendation_System_New_v3.ipynb` | Source of synthetic query personas/templates referenced by notebook 01. |
| `archive/legacy_notebooks/` | Older standalone implementations retained for reference. |
| `data/policies.csv` | Current exported policy catalogue consumed by the pipeline. Notebook 01 rewrites it from its current contents plus the root eligibility source. |
| `data/source/eligibility_from_brochures.csv` | Brochure eligibility input merged by notebook 01. It includes official LIC product and brochure URLs. |
| `labels/relevance_labels.csv` | Synthetic-only training and smoke-test labels. |
| `data/source/human_eval_queries.csv`, `data/source/human_labels.csv` | Separate evaluation-only queries and user-confirmed human-reviewed grades. |
| `data/source/policies.csv` | Original catalogue CSV from the source upload; retained for reference, not used as the current exported catalogue. |
| `relevance_labels_TEMPLATE.csv` | Unscored label template, remapped to current policy IDs by notebook 01. |
| `artifacts/` | Notebook handoffs: synthetic and human query/intent files, catalogue parquet, split manifest, GraphML, separate label parquet files, and model/retrieval caches. |
| `results/` | Catalogue audit, data status, fine-tuning status, synthetic retrieval smoke output, human-evaluation candidates/results, selected weights, and evaluation status. |
| `tools/policy_eligibility.py` | Shared deterministic eligibility gate used by notebooks 02 and 04. |
| `tools/build_synthetic_smoke_labels.py` | Rebuilds synthetic smoke labels from the template and current catalogue. |
| `tools/run_notebooks.py` | Executes notebooks 01–04 sequentially. |
| `tests/` | Data, gate/parser, training, and retrieval regression checks. |
| `archive/` | Superseded catalogue snapshots, legacy notebooks, and prior exports retained for reference. |
| `artifacts/model/` | Fine-tuned model checkpoints and manifest; runtime caches, not evidence of model quality. |

## 4. Configuration

`config.json` currently sets:

| Setting | Meaning |
|---|---|
| `random_seed: 42` | Seed used in data splits and training. |
| `smoke_test_mode: true` | Labels and reports are identified as smoke-test outputs. |
| `allow_model_downloads: true` | Allows Sentence Transformers models to download if not cached; an internet connection may be needed on first run. |
| `enable_synthetic_smoke_training: true` | Enables the demonstration fine-tuning stage. |
| `source_notebook` | Notebook 01 reads synthetic persona templates from this file. |
| `real_labels_path` | Currently points at `labels/relevance_labels.csv`, which contains synthetic labels in this repository. This name does not mean human labels are supported. |
| `embedding_model_name` | Base encoder: `all-MiniLM-L6-v2`. |
| `cross_encoder_model_name` | Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`. |
| `display_recommendations: 8` | Maximum policies shown in each response; every eligible policy is ranked before display truncation. |

The notebook setup cells check/install some missing Python packages; the active stack includes pandas, NumPy, PyArrow, scikit-learn, NetworkX, PyTorch, and Sentence Transformers. FAISS is optional in the retrieval path. Model downloads and GPU availability affect runtime but should not alter the safety gate.

## 5. Data contracts

### 5.1 Policy catalogue

`data/policies.csv` currently has 47 unique rows. `policy_id` is `LIC_<plan_number>`; the UIN is stored separately. Current status distribution is 40 active, 5 withdrawn, and 2 unresolved. The 2 unresolved policies (Index Plus 873 and New Pension Plus 867) are outside the recommendation pool. Arogya Sanjeevani 906 and superseded plan 854 are absent.

Important fields:

- Identity and lifecycle: `policy_id`, `policy_name`, `plan_number`, `uin`, `status`, `supersedes`, `withdrawn_versions`, `statutory_status`.
- Eligibility: `min_age`, `max_age`, `min_age_days`, `min_entry_age_days`, `min_maturity_age`, `max_maturity_age`, `min_term`, `max_term`, `premium_paying_term`, `min_sum_assured`, `max_sum_assured`, `max_sum_assured_unlimited`, `rules_json`, `eligible_gender`.
- Verification/provenance: `verified`, `identity_verified`, `age_verified`, `term_verified`, `sum_assured_verified`, `category_verified`, `verified_scope`, `verification_status`, `source_url`, `policy_url`, `brochure_url`, source notes.
- Retrieval hints: `category`, `financial_goal`, `risk_tolerance_inferred`, `premium_level_inferred`, `coverage_type`, and `description`.

The explicit old-to-current mappings are applied by notebook 01. For example, 936 maps to current 736; 854 maps to current 954. The current plan number is the recommendation ID; historical numbers remain in `supersedes` and `withdrawn_versions`.

### 5.2 Brochure eligibility input

`data/source/eligibility_from_brochures.csv` has 16 plan rows, all currently marked `eligibility_verified=True`. Notebook 01 merges by `plan_no` and rejects UIN mismatches. It copies eligibility fields from the matched row; `verified` and `verified_scope` are set only when `eligibility_verified=True`. The source was updated on 19 November 2026 to include brochure-based limits for plans 954, 875, 876, 877, and 878. The merge trusts those flags; it does not independently inspect or validate the underlying brochure text.

The file includes official LIC product-page and brochure URLs, but the latest pipeline did not independently re-extract every value from those documents. “Verified” reflects the supplied source flag and scope; it is not an independent audit performed by the notebook.

### 5.3 Queries, labels, and splits

- `artifacts/queries.csv`: 18 synthetic queries with `query_id`, `group_id`, `raw_query`, `persona`, and `provenance`.
- `labels/relevance_labels.csv`: 324 rows with `query_id`, `group_id`, `policy_id`, `grade`, and `annotator`. All current rows use `SYNTHETIC_SMOKE_TEST`; grades are 0–3.
- `data/source/human_eval_queries.csv`: separate evaluation prompts. Their provenance is unspecified in the current source file. No human labels are used to train or tune the model.
- `data/source/human_labels.csv`: user-confirmed, human-reviewed grades (0–3) for a subset of those prompts. The source does not identify individual reviewers; normalized artifacts record `HUMAN_UNSPECIFIED` rather than inventing an identity.
- `artifacts/human_eval_labels.parquet`: validated human labels with `HUMAN_REVIEWED_BY_USER` provenance; kept separate from synthetic labels.
- `relevance_labels_TEMPLATE.csv`: 324 ungraded query-policy rows.
- `artifacts/splits.json`: group-disjoint synthetic train/validation/test query splits plus a policy-held-out split. Human evaluation queries are not included in training splits.

## 6. Eligibility and intent behavior

`tools/policy_eligibility.py` is the canonical gate. It returns `eligible`, `ineligible`, or `unknown`, plus reasons. Retrieval in notebook 04 removes every policy that is not `eligible` before ranking.

The gate:

1. Rejects withdrawn plans; unresolved/out-of-pool plans remain unknown.
2. Evaluates verified entry-age bounds, including term-specific and PPT-specific age rules.
3. Checks requested term bounds and any allowed-term rule.
4. Checks `age + term` against maximum maturity age when that maximum is available. If no maximum maturity age is needed or supplied, that alone does not force `unknown`.
5. Checks requested sum assured against configured minimum and maximum, including unlimited maximum cases.
6. Checks gender only when `eligible_gender` is populated. Blank/NaN means no gender condition is configured. The current source marks Bima Lakshmi (881) as women-only (`eligible_gender=female`), based on user confirmation from the brochure cover.
7. Treats blank/NaN `rules_json` as no additional rules; malformed or non-object JSON is an error.
8. For `by_ppt` rules, uses an explicitly supplied PPT only. If PPT is omitted, it evaluates every listed PPT option: any passing option means eligible; all definite failures mean ineligible; otherwise the result is unknown. Missing customer age remains unknown.

The intent parser extracts customer/child age, policy term, PPT, sum assured expressed in lakh/crore, goals, risk/premium preferences, and a small set of contradictions. It is regex-based; it is not a conversational language model and does not reliably support arbitrary phrasing, multilingual inputs, or nuanced financial intent.

## 7. Retrieval and ranking

Notebook 04 builds policy text, filters every catalogue policy through the shared eligibility gate, ranks the full eligible set (at most 16), and limits only the displayed recommendations to eight. Results include the eligible candidate count and how many eligible policies were omitted from the display. It combines:

- Dense bi-encoder similarity (base or fine-tuned model).
- BM25 lexical relevance.
- KG preference features for goal, risk, and premium.
- Cross-encoder pair scores when available; otherwise dense ranking fallback.

Signals are min-max normalized; when a vector is constant, tied values receive neutral `0.5`. Ties are ordered deterministically by policy ID. The final score is a ranking score, **not** a probability.

The smoke output is a wiring check, not a relevance finding. When normalized signals tie, they receive neutral `0.5`; deterministic policy-ID ordering resolves remaining ties. The selected synthetic validation weights should not be treated as evidence of ranking quality.

## 8. Training and evaluation

Notebook 03 creates `(query, positive policy, negative policy)` triplets from synthetic grades and trains a small MNRL fine-tuning run. It asserts synthetic provenance, rejects nonzero human-label counts, and caches checkpoints under `artifacts/model/`. The fine-tuning run is functional smoke evidence only.

Notebook 04 sweeps alpha/beta combinations on validation profiles and evaluates on held-out profiles. The current result file marks held-out ranking evaluation `NOT_EVALUABLE_NO_ELIGIBLE_RELEVANT_JUDGMENTS`. Its zeros must not be described as measured ranking performance. Current labels and query goals are synthetic and partly constructed from policy categories; they are not user or advisor judgments.

### Human-reviewed evaluation is separate from training

Notebook 01 validates `data/source/human_labels.csv` against `data/source/human_eval_queries.csv`, checks IDs/groups/grades, and writes a separate human-label artifact. Notebook 02 parses those queries. Notebook 04 applies fixed weights selected on synthetic validation data and evaluates only eligible judged policies; unjudged policies are excluded rather than presumed irrelevant. Human labels are not used for training or hyperparameter tuning. Results are saved to `results/human_evaluation.json` and ranked candidates to `results/human_evaluation_candidates.csv`.

The current query file has no provenance column, so the pipeline records query provenance as `UNSPECIFIED`. Reviewer identity is not present in the source; artifacts say `HUMAN_UNSPECIFIED`. The judgments were treated as human-reviewed based on the user's confirmation. This is a separate evaluation path, not evidence that the recommender is production-ready or suitable for financial advice. Incomplete judgments limit what can be inferred from rankings.

## 9. What has been done

- Rebuilt the catalogue to current `LIC_<plan_number>` IDs and retained the supplied supersedes history.
- Removed plan 854 in favor of plan 954; kept 867 and 873 unresolved/out of the candidate pool; removed 906.
- Added eligibility fields and official LIC page/brochure URLs; merged the supplied 16-row source by plan number with UIN validation.
- Added shared eligibility rules for age, term, maturity, sum assured, optional gender, PPT options, and safe handling of blank/NaN rules.
- Added requested regression coverage for Jeevan Labh, Bima Kavach, Yuva Term, Nav Jeevan Shree, Bima Platinum, Bima Lakshmi, and blank rules/gender cases.
- Fixed normalization of tied ranking signals so ties receive neutral `0.5`, rather than `0.0`.
- Removed duplicate `eligibility_*` catalogue columns and updated `verification_status` for the 16 supplied rows now marked verified.
- Moved old catalogue snapshots and unused additions into `archive/`.
- Kept synthetic training labels separate from the user-confirmed human-reviewed evaluation labels.
- Added separate human-query intent parsing and evaluation artifacts; human labels are excluded from model training and tuning.
- Executed notebooks 01–04 with the separate evaluation path; the existing synthetic smoke outputs remain synthetic-only.

## 10. What should be done next

### Priority 0: establish trustworthy policy facts

1. Independently check every “verified” value against the linked current LIC brochure; record page/section and review date.
2. Resolve the duplicate eligibility-source location: make the root file the single source of truth or change the notebook to one canonical path.
3. Add a catalogue validation report for missing source URLs, UIN conflicts, stale plan versions, and inconsistent verification flags.

### Priority 1: support real evaluation safely

1. Record reviewer identity/adjudication details where appropriate; the current labels do not identify individual annotators.
2. Expand label coverage to more policies and queries; keep unjudged items distinct from irrelevant ones.
3. Collect representative real customer queries and independent advisor-reviewed grades before making claims about real-world recommendation quality.
4. Revisit ranking-score ties and tune only on a designated human-labeled validation set after enough independent judgments exist.

### Priority 2: make the system usable

1. Add a small API or user interface around intent extraction, missing-field clarification, eligibility filtering, and recommendation explanation.
2. When gate status is unknown, ask for the missing fact rather than recommending the policy as eligible.
3. Add operational logging, error handling, accessibility, privacy controls, and an explicit financial-advice disclaimer before any deployment.

### Priority 3: improve maintainability

1. Add a pinned dependency manifest and clean-environment CI for notebook execution and tests.
2. Make notebook runs deterministic and avoid unnecessary model downloads in routine data/gate tests.
3. Clearly label/archive the older standalone notebooks so readers do not confuse them with the current pipeline.
4. Refresh LIC source data on a documented schedule and rerun schema, URL, and eligibility checks after each update.

## 11. Operational cautions

- Do not use the current synthetic ranking output as a policy recommendation or quality benchmark.
- A model checkpoint is only a cached training artifact; its existence does not imply validated performance.
- The catalogue is not an underwriting engine. It does not calculate premiums, assess medical history, provide riders, or guarantee acceptance.
- LIC pages and brochure content can change. Verify current official policy documents before making decisions.
- This project is educational decision support, not financial advice.
