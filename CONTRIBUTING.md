# Contributing to ChipClaw

Useful contributions include reproducible condition-binding failures, numerical/unit edge cases, table-layout regressions, clearer replay examples and documentation improvements.

Start with the lightweight workbench instructions in [README.en.md](README.en.md). Its dependencies and saved runs do not require downloading models. The repository's CI covers that lightweight contract scope; full ranking or GPU experiments have separate prerequisites in [PUBLICATION.md](PUBLICATION.md).

For an issue, include the source manual and official URL, printed/PDF page numbering, question, expected condition/role binding, observed result, software version and whether it was a saved replay or a live request. Use the smallest attributable source excerpt needed to reproduce the problem. Do not upload private documents, credentials or full copyrighted manuals.

Keep frozen records and negative outcomes intact. New training/evaluation should use a new report directory, identify sampling and checkpoint selection, and distinguish reused weak labels from independent human judgments. Ranking MRR must not be presented as generated-answer accuracy; unknown verification outcomes must not be silently changed to supported.

Small patches with a concrete before/after reproducer are easiest to review. Keep heavyweight model/PDF downloads out of lightweight CI and maintain the distinction between source provenance and semantic correctness.
