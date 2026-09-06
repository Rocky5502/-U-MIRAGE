# Contributing

U-MIRAGE is a research codebase. Changes must preserve reproducibility and the strict separation between synthetic validation and scientific evidence.

- Add deterministic tests for new uncertainty signals, dataset adapters, or model adapters.
- Never commit third-party benchmark media unless its license explicitly permits redistribution.
- Freeze exact model and dataset revisions for paper runs.
- Never hand-edit paper numbers; generate tables/figures from raw immutable artifacts.
- New baselines must record source/version/configuration and all generation parameters.
- `artifacts/smoke/` is plumbing only and must never enter paper aggregation.
- Avoid hidden network calls in library code; downloads belong in explicit preparation scripts.
