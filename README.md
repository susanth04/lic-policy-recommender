# LIC Policy Recommender

## Architecture

The workflow is a sequence of Colab-compatible notebooks with file-based handoffs:

- `01_data_and_catalogue.ipynb` exports the reference notebook catalogue and synthetic queries, inspects labels, and creates query-group and policy-held-out splits.
- `02_intent_and_kg.ipynb` parses intent, builds the GraphML knowledge graph, and enforces age and requested-term eligibility gates. Missing age is `unknown`, never eligible.
- `03_finetune_biencoder.ipynb` contains seeded, cached MNRL training paths, gated on human grades and explicit model-download permission.
- `04_retrieval_fusion_rerank.ipynb` exercises gated retrieval and fallback ranking locally. It does not tune weights on synthetic data.

Notebooks exchange files through `artifacts/` and `results/`; they do not depend on in-memory state from another notebook. The catalogue CSV preserves the source verification wording as `claimed_verification`; all exported catalogue records have `verified=False`.

## Reproduce and test

Run `make reproduce` to execute notebooks 01–04 in order, or run each notebook in Colab. Use `make test` for the smoke tests. On Windows, set the Python launcher explicitly if needed, for example `make reproduce PYTHON="py -3.11"`.

The current label file is a template with no grades or annotators. Consequently, `results/data_status.json` records label availability, `results/finetune.json` records the MNRL skip, and `results/weights_smoke.json` records that alpha, beta, and KG preference weights were not selected. `results/retrieval_smoke.json` contains only script-generated wiring and eligibility checks. No retrieval-quality metrics or tuned weights are claimed.

Evaluation and report-generation notebooks 05–06 are intentionally deferred until human-graded relevance labels exist. After labels are supplied at `labels/relevance_labels.csv`, rerun the data notebook and tests before enabling training or validation-based tuning.

## Limitations and disclaimer

Queries are synthetic and are for smoke testing only. Catalogue facts are unverified; any source verification text is a claim copied from the prototype, not an independently checked citation. The system does not perform medical underwriting or calculate live premiums.

This is educational decision support, not financial advice. It is not a substitute for official policy documents or guidance from a licensed insurance advisor.
