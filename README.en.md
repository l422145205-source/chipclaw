# ChipClaw

**Condition-aware chip-manual retrieval, domain ranking and numerical evidence checks.**

[中文](README.md) · [Reproduction and resources](PUBLICATION.md) · [Contributing](CONTRIBUTING.md)

Finding the correct number in a datasheet is only part of the task: its unit, MIN/TYP/MAX role, operating condition and table context must stay attached. ChipClaw retrieves source evidence, reconstructs bounded table context and checks numerical claims against physical PDF rows.

## Try the saved-run workbench without models

```bash
git clone https://github.com/l422145205-source/chipclaw.git
cd chipclaw
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-web.txt
python -m uvicorn chipclaw.workbench:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The lightweight mode displays 28 saved actual runs across 20 manuals. It is replay, not new retrieval or newly generated answers. A static replay is also available with `python -m http.server 8000 --directory web`.

The workbench exposes original snippets, physical table context, source-page links, provenance and supported/contradicted/unknown numerical checks. Online `/evidence` requires the full model, index and PDF environment described in [PUBLICATION.md](PUBLICATION.md).

## Implementation

- **Domain ranking:** full-parameter multilingual MiniLMv2 training with condition/role-conflict negatives; controlled no-KL/KL comparisons; Qwen3-Reranker-0.6B ranking LoRA using yes-minus-no logits.
- **Retrieval:** protected model-number/value/unit rewriting, BM25 multi-view RRF, student or domain-adapted reranking, and optional fusion on a fixed visual subset.
- **Table evidence:** source PDF geometry, bounded dependency closure over headers/rows/nearby notes, and multi-start selection under character and page budgets.
- **Independent verification:** deterministic binding of parameter, value, unit, MIN/TYP/MAX and conditions. Unknown means the rule-based verifier did not determine the claim.

The conservative teacher-escalation gate failed calibration and currently falls back to the student. The negative result is retained. This is not a claim of a successful adaptive cascade.

## Frozen experiments

Corpus: 20 official manuals, 1,822 pages and 10,668 text chunks. Visual experiments use a fixed 240-page subset.

| Controlled ranking on 1,042 reused bilingual question forms | MRR@10 |
|---|---:|
| Multilingual MiniLMv2 base | 0.8694 |
| Full-parameter domain training, no KL, selected on dev | 0.9606 |
| Matched KL comparison | 0.9579 |
| Qwen3 domain ranking LoRA | 0.9686 |

On a different experiment with actual full-text recall candidates—240 question forms, 16 same-document candidates—the student reached source-page MRR 0.8973 and the all-candidate domain LoRA 0.9531. Reported mean ranking times were 14.7 ms and 355.9 ms on the original machine, excluding model loading and retrieval. Do not compare these directly with the controlled-candidate table.

Labels are source-constrained weak labels/template questions; the holdout has already been reused, and there is no independent human fact gold set. MRR measures ranking, not generated-answer accuracy. LoRA adapts the ranker, not answer generation. See [frozen configurations and outcomes](experiments/README.md).

## Read the code

| Concern | Entry point |
|---|---|
| Lightweight workbench | [workbench.py](chipclaw/workbench.py), [web/](web) |
| Ranking and cascade | [retrieval_v4.py](chipclaw/retrieval_v4.py), [cascade_v4.py](chipclaw/cascade_v4.py) |
| Physical table packets | [packets_v4.py](chipclaw/packets_v4.py) |
| Numerical verification | [fact_verifier.py](chipclaw/fact_verifier.py) |
| Domain training | [MiniLM training](scripts/train_multilingual_v4.py), [LoRA training](scripts/train_domain_lora_v4.py) |

Public CI runs lightweight workbench/data contracts without model downloads; it is not full model-training validation. Local and public test scopes are distinguished in the [Chinese guide](README.md). Full PDFs, model weights, personal documents and environment caches are excluded from this repository. See [third-party provenance](THIRD_PARTY.md).

This is a research and engineering prototype. Independent answer evaluation, broader verification coverage and service governance remain open. Code development uses AI assistance.
