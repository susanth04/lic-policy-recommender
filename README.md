# LIC Policy Recommender

For detailed architecture, data contracts, eligibility logic, current limitations, completed work, and a prioritized roadmap, see [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md).

## Architecture

The workflow is a sequence of Colab-compatible notebooks with file-based handoffs:

- `01_data_and_catalogue.ipynb` loads `data/policies.csv`, extracts synthetic queries from the reference notebook, remaps synthetic labels, and creates query-group and policy-held-out splits.
- `02_intent_and_kg.ipynb` parses synthetic and human-evaluation queries, builds the GraphML knowledge graph, and enforces brochure-based age, maturity, term, and sum-assured eligibility gates. Missing required inputs or unchecked limits are `unknown`, never eligible.
- `03_finetune_biencoder.ipynb` runs seeded, cached MNRL triplet fine-tuning on explicitly marked synthetic smoke labels.
- `04_retrieval_fusion_rerank.ipynb` selects weights using synthetic validation data and separately evaluates the fixed system on user-confirmed human-reviewed grades.

Notebooks exchange files through `artifacts/` and `results/`; they do not depend on in-memory state from another notebook. Catalogue IDs are `LIC_<plan_number>`. The catalogue records current LIC category-page listings, `supersedes` and `withdrawn_versions`; the source page and its displayed update date are retained per row. `data/source/eligibility_from_brochures.csv` is merged by `plan_no`; `verified` and `verified_scope` are copied only for brochure rows marked `eligibility_verified=True`. Index Plus 873 and New Pension Plus 867 are unresolved and excluded from the recommendation pool.

Retrieval filters on the eligibility gate, ranks every eligible policy (at most 16), then displays up to eight. The result JSON reports eligible candidate counts and the number omitted from display.

`data/policies.csv` is the input used for current exports. Superseded catalogue snapshots and unused additions are preserved under `archive/`. Withdrawn products are ineligible, unresolved or unverified limits stay `unknown`. Brochure eligibility is merged from the root CSV; keep its verification flags and source notes current.

## Reproduce and test

Run `make reproduce` to execute notebooks 01–04 in order, or run each notebook in Colab. Use `make test` for the smoke tests. On Windows, set the Python launcher explicitly if needed, for example `make reproduce PYTHON="py -3.11"`.

`labels/relevance_labels.csv` contains synthetic smoke-test grades marked `SYNTHETIC_SMOKE_TEST`; `grade` is the relevance column (0–3). Notebook 03 trains only on these synthetic labels. `data/source/human_labels.csv` and `data/source/human_eval_queries.csv` feed a separate evaluation-only path: labels are not used for training or tuning. Query provenance is unspecified in the current CSV. Outputs are `results/human_evaluation.json` and `results/human_evaluation_candidates.csv`; neither establishes real-world recommendation quality or KG benefit.

The current held-out profile has no grade-2/3 judgments for policies that pass the hard eligibility gate. Accordingly, `results/synthetic_smoke_evaluation.json` marks its ranking metrics `NOT_EVALUABLE_NO_ELIGIBLE_RELEVANT_JUDGMENTS`; the resulting zero scores must not be interpreted as measured ranking quality.

Evaluation and report-generation notebooks 05–06 remain deferred. Keep synthetic training labels and human-reviewed evaluation labels separate; update the human query/label CSVs together and rerun notebooks and tests after changes.

## Limitations and disclaimer

Queries are synthetic and are for smoke testing only. Catalogue identity/status comes from LIC listings; eligibility facts are only marked verified where the supplied brochure eligibility file says they were checked. The system does not perform medical underwriting or calculate live premiums.

This is educational decision support, not financial advice. It is not a substitute for official policy documents or guidance from a licensed insurance advisor.
