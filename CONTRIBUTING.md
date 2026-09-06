# Contributing

Research changes must preserve reproducibility and the distinction between synthetic validation and scientific evidence.

- Add deterministic tests for new uncertainty signals or adapters.
- Never commit benchmark media unless its license explicitly permits redistribution.
- Freeze model and dataset revisions for paper runs.
- Do not hand-edit aggregated paper numbers; generate them from raw prediction artifacts.
- New baselines must record exact source/version/configuration.
- `artifacts/smoke` is plumbing only and must never enter paper aggregation.
