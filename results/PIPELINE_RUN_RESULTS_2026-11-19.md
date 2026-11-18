# Pipeline Run Results — 19 November 2026

## Run status

- `python tools\run_notebooks.py`: notebooks 01–04 completed successfully.
- `python -m pytest -q`: 19 tests passed; 3 upstream deprecation warnings.
- Catalogue export: 47 policies; 16 brochure rows marked verified.
- Synthetic pipeline: 18 smoke queries across 6 groups, using 324 synthetic labels. Synthetic labels remain the only training labels.

## Human-evaluation data

- 60 evaluation queries across 12 groups.
- 549 user-confirmed human-reviewed grades across 55 queries; `HQ_055`–`HQ_059` remain ungraded.
- 41 queries had at least one positively graded policy pass the eligibility gate. The other 19 had no eligible relevant judgment, so they are not evaluable.
- No eligibility violations were reported in the candidate results.
- Human labels were not used for model training or tuning. Ranking weights came from synthetic validation data.
- Query provenance and individual reviewer identity are unspecified in the supplied files.

## Outputs

- Human-evaluation summary: [human_evaluation.json](./human_evaluation.json)
- Ranked candidates with grades: [human_evaluation_candidates.csv](./human_evaluation_candidates.csv)
- Human-label input status: [human_evaluation_data_status.json](./human_evaluation_data_status.json)
- Synthetic evaluation status: [synthetic_smoke_evaluation.json](./synthetic_smoke_evaluation.json)

No ranking-quality metric values are reported here as findings. The human-evaluation data is incomplete, and its queries have unspecified provenance; results should not be taken as evidence of real-world recommendation quality or production readiness.
