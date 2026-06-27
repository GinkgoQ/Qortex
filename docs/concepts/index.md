# Concepts

Qortex is built around a few design decisions that affect how you use it. Understanding them makes the API predictable.

## Core ideas

[**What is Qortex**](what-is-qortex.md) — Qortex is not a downloader. It is a decision layer that sits between OpenNeuro and your training pipeline. It tells you whether a dataset is usable before you transfer any files.

[**Readiness first**](readiness-first.md) — Every dataset has a readiness state: not inspected, inspected (manifest only), downloaded, validated, and conversion-ready. Qortex checks readiness at each stage and tells you what is blocking progress.

[**OpenNeuro and BIDS**](openneuro-and-bids.md) — How Qortex uses the OpenNeuro GraphQL API, what a snapshot is, how BIDS structure maps to Qortex entities. Covers what data the API exposes: metadata, engagement, demographics, README, BIDS validation.

[**Data model**](data-model.md) — The types that carry information through the pipeline: `RichDatasetInfo`, `SnapshotSummary`, `DatasetEngagement`, `Dataset`, `Manifest`, `FileRecord`, `DatasetProfile`, `Artifact`.

[**Workflow**](workflow.md) — The full pipeline from OpenNeuro catalog search to a trained model, with decision points at each stage.
