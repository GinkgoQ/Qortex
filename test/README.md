# Qortex staged scenario projects

These are intentionally plain Python scripts, not pytest tests. Each numbered
directory is a small runnable project that imports the installed `qortex`
package, prints the workflow output, and then checks that the output is
coherent.

Run all projects:

```bash
python test/run_all.py
```

Run one project:

```bash
python test/5_eda_events/run.py
```

The stages move from install/configuration to manifest semantics, structural
planning, remote metadata preview, metadata-only and exact-path plans, EDA,
event/table conversion, readiness, loaders, windowing, local indexing,
validation report exports, catalog search, CLI behavior, high-level Dataset
facade usage, live OpenNeuro metadata, decision-first workflows, and deep
catalog metadata ingestion when the network is available.

`test/run_all.py` shares one real metadata download across downstream projects.
If the official `bids-validator` CLI is not installed, the validation scenario
prints that dependency state explicitly and does not fabricate validation
results.
