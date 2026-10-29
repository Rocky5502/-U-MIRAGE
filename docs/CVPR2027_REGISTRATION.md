# CVPR 2027 — U-MIRAGE registration draft

> **Status:** Registration draft only. This file does not mean the paper has been enrolled or submitted to OpenReview. Confirm all metadata and author enrollment before the November 10, 2026 AoE deadline. The manuscript is not yet submission-ready.

## Title (OpenReview)

U-MIRAGE: Source-Aware Uncertainty Propagation for Reliable Multi-Image Vision-Language Reasoning

## Abstract (OpenReview)

Multi-image vision-language models can produce the same incorrect answer for different reasons: ambiguous visual evidence, failed integration across images, or erroneous downstream reasoning. We introduce U-MIRAGE, a source-aware uncertainty framework that separates these stages, propagates their signals to answer-level risk, and maps the dominant signal to a corrective action. We also introduce MIRAGE-U, a matched-intervention protocol for testing whether estimated uncertainty sources respond selectively to targeted perturbations. Our development-stage evaluation reveals substantial limitations. Across 360 matched construction-intent comparisons, visual and reasoning selectivity are negative, while integration selectivity is indistinguishable from zero. On 128 natural development episodes, uncertainty propagation does not improve AUROC or AURC over a baseline using identical source signals. A matched-budget repair study is non-diagnostic for comparative efficacy because all nonzero-budget policies score zero under the frozen exact-match criterion, indicating a possible output-contract or evaluation failure. Feature-enrichment effects across four local VLM families are heterogeneous. Two hosted multimodal endpoints fail a common uncertainty-interface usability gate, while a separate exploratory repair-transfer study yields inconclusive, model-dependent effects. These findings highlight the importance of falsifiable source attribution, matched-signal controls, and output-contract validation when evaluating uncertainty-aware multimodal systems. Human source validation and final holdout evaluation remain incomplete.

## Suggested form fields

- **Subject area:** Multimodal learning (select the closest exact available option).
- **Keywords:** multi-image reasoning, vision-language models, uncertainty estimation, uncertainty propagation, hallucination detection, selective verification
- **Student paper:** Yes, provided the first author remains a student at submission.
- **Authors:** Enter all genuine authors, matching their complete OpenReview profiles. Do not infer a list from manuscript acknowledgements or repository contributors.
- **External links confirmation:** Check only after reviewing the actual submission and supplement. Remove links that expand content, compromise anonymity, or subvert reviewing.

## Official deadlines

- **Paper registration and all author enrollment:** November 10, 2026 (AoE).
- **Full manuscript:** November 16, 2026 (AoE).
- **Supplementary material:** November 23, 2026 (AoE).

## Mandatory submission gates — do not declare ready until all pass

- [ ] Use the official CVPR 2027 template and keep **main technical content within eight pages**. References and the required LLM Usage Statement may occupy additional pages.
- [ ] Check source-to-PDF page boundaries and legibility; the earlier v33 PDF was 15 pages total.
- [ ] Include an accurate, anonymous **LLM Usage Statement** identifying tools and affected research/writing components.
- [ ] Complete **all authors' OpenReview profiles and enrollment forms** by Nov 10; the author set cannot be added to or reduced after this deadline.
- [ ] Check reciprocal-reviewer eligibility and submission quota from the actual author list and their OpenReview publication records.
- [ ] Keep the PDF and supplement self-contained, anonymized, and free of disallowed external links.
- [ ] Verify citations, data licenses, exact model identifiers, source partitions, and code/result provenance.
- [ ] Audit the **RQ3 all-zero repair result** for output parsing and execution artifacts without tuning policies based on the observed outcomes; do not present it as evidence of actual repair harm.
- [ ] Respect the frozen RQ1 negative results, RQ2 null results, hosted estimator-gate failures, and RQ4c exploratory confidence intervals.
- [ ] Do not call a development result a protected final result; **Calibration-60 and Holdout-360 remain sealed**.
- [ ] Confirm no conflicting or substantially overlapping submission violates dual-submission rules.
- [ ] All authors approve submitted contents, metadata, and claims.

## Sources and provenance

- Official guidelines: https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines
- Official schedule: https://cvpr.thecvf.com/Conferences/2027/Dates
- Call for Papers: https://cvpr.thecvf.com/Conferences/2027/CallForPapers
- Submission portal: https://openreview.net/group?id=thecvf.com/CVPR/2027/Conference
- Manuscript basis: U-MIRAGE v33 PDF; result status corroborated in repository README and docs/RESULTS.md.
- **Never upload raw credentials, private compute paths, test/holdout labels, or author-identifying paper assets to anonymous submission material.**
