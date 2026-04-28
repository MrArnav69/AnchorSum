<p align="center">
  <h1 align="center">AnchorSum</h1>
  <p align="center">
    <strong>Verifier Exploitation in NLI-Guided Iterative Refinement:<br/>A Controlled Empirical Analysis</strong>
  </p>
  <p align="center">
    Arnav Gupta · Independent Researcher, Nepal
  </p>
  <p align="center">
    <a href="#installation">Installation</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#repository-structure">Structure</a> ·
    <a href="#reproducing-results">Reproduce</a> ·
    <a href="#results-summary">Results</a> ·
    <a href="#citation">Citation</a>
  </p>
</p>

---

## Overview

AnchorSum is a modular, training-free pipeline for faithful multi-document summarization. It combines entity-guided anchor extraction, anchor-conditioned draft generation, dual-mode faithfulness auditing (sentence-level NLI + entity hallucination filtering), and flag-guided iterative revision — all without modifying any model weights.

This repository accompanies the paper *"Verifier Exploitation in NLI-Guided Iterative Refinement: A Controlled Empirical Analysis"*, which uses AnchorSum as a controlled experimental setting to demonstrate that verifier exploitation — satisfying an NLI auditing metric while degrading real faithfulness — manifests even in zero-gradient, prompt-only refinement loops.

**Key findings:**
- AnchorSum (T<sub>max</sub>=1) achieves a **6.3% relative SummaCConv inconsistency reduction** over its unaugmented base (p=4.49×10⁻²⁸)
- A second revision cycle inflates SummaCConv by +0.185 while collapsing BARTScore s→d by −2.566 nats — **Wilcoxon W=0** across all 498 instances — confirming verifier exploitation
- Outperforms PEGASUS-MultiNews, PRIMERA-MultiNews, and BART-large-CNN across all dimensions in dual-judge LLM-as-judge evaluation (jury score: 8.30/10)

---

## Architecture

```
Source Corpus D = d₁ ‖ … ‖ dₖ
         │
         ▼
┌─────────────────────────────────────┐
│  PHASE 0 — Anchor Extraction        │
│  spaCy en_core_web_trf (RoBERTa)    │
│  Top-15 entities by corpus frequency │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PHASE 1 — Draft Generation         │
│  Meta-Llama-3.1-8B-Instruct         │
│  Greedy decoding · rep. penalty 1.1 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 2 — Dual-Mode Faithfulness Audit                 │
│                                                         │
│  ┌───────────────────────────┐  ┌─────────────────────┐ │
│  │  2a: NLI Audit            │  │  2b: Entity Guard   │ │
│  │  nli-deberta-v3-large     │  │  Coverage + extrinsic│ │
│  │  435M · SNLI + MultiNLI   │  │  hallucination check │ │
│  │  F_NLI = {non-entailed}   │  │  F_ENT = F_cov∪F_hal│ │
│  └─────────────┬─────────────┘  └──────────┬──────────┘ │
│                └──────────┬────────────────┘            │
│                           │ F = F_NLI ∪ F_ENT           │
└───────────────────────────┼─────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               │                         │
           F = ∅?                    F ≠ ∅ and i < T_max
               │                         │
               ▼                         ▼
         Final Ŝ*             ┌────────────────────────┐
                               │  PHASE 3 — Revision    │
                               │  LLM expert-editor role│
                               │  Flags as numbered list│
                               │  → loops back Phase 2  │
                               └────────────────────────┘

  Default: T_max = 1 (recommended; T_max=2 triggers verifier exploitation)
```

---

## Repository Structure

```
AnchorSum/
│
├── src/                                  # Core pipeline source code
│   ├── __init__.py
│   ├── pipeline.py                       # Main orchestrator — composes Phases 0–3
│   ├── llm_summarizer.py                 # LLaMA draft generation + revision prompting
│   └── verification/
│       ├── __init__.py
│       ├── nli_verifier.py               # DeBERTa-v3-Large sentence-level NLI audit
│       └── entity_guard.py               # spaCy anchor extraction + hallucination filter
│
├── ablations/                            # Experiment runners
│   ├── ablation_base_runner.py           # Shared base: dataset loading, checkpointing
│   ├── Component_Ablation/
│   │   └── run_all_sequential.py         # Runs all 4 configs × 500 samples
│   └── Revision_Depth/
│       └── revision2.py                  # Full pipeline at T_max=2
│
├── scripts/                              # Per-metric evaluation scripts
│   ├── Component_Ablation/
│   │   ├── evaluate_summac_final.py                # SummaCConv (ViT-C)
│   │   ├── evaluate_alignscore_simple.py           # AlignScore (RoBERTa-Large)
│   │   ├── evaluate_bartscore_simple.py            # BARTScore (BART-Large-CNN)
│   │   ├── evaluate_rouge_bertscore_simple.py      # ROUGE-1/2/L + BERTScore
│   │   ├── evaluate_bertscore_xlarge.py            # BERTScore (DeBERTa-XLarge-MNLI)
│   │   └── evaluate_unieval_fluency_simple.py      # UniEval fluency
│   └── Revision_Depth/
│       ├── evaluate_summac_full_revisions_2.py
│       ├── evaluate_alignscore_full_revisions_2.py
│       ├── evaluate_bartscore_full_revisions_2.py
│       ├── evaluate_rouge_bert_full_revisions_2.py
│       └── evaluate_unieval_fluency_full_revisions_2.py
│
├── data/
│   ├── multi_news_500_samples.json       # Cached 500-sample subset (seed=42)
│   ├── document.json                     # Sample outputs with multi-model comparisons
│   └── ablations/                        # Generated summaries per configuration
│       ├── base/                         #   No auditing, no revision
│       ├── no_nli/                       #   Entity Guard only
│       ├── no_entity/                    #   NLI audit only
│       ├── full/                         #   Full pipeline (T_max=1)
│       └── full_revisions_2/             #   Full pipeline (T_max=2)
│
├── Results/                              # Precomputed metric outputs (CSV)
│   ├── Component Ablation/
│   │   ├── summac_final_results/
│   │   ├── alignscore_results/
│   │   ├── bartscore_simple_results/
│   │   ├── bertscore_xlarge_results/
│   │   ├── rouge_bert_simple_results/
│   │   └── unieval_fluency_results/
│   └── Revision Depth/
│       ├── summac_full_revisions_2_results/
│       ├── alignscore_full_revisions_2_results/
│       ├── bartscore_full_revisions_2_results/
│       ├── rouge_bert_full_revisions_2_results/
│       └── unieval_fluency_full_revisions_2_results/
│
├── Significance_Testing/
│   └── wilcoxon_results.csv              # Paired Wilcoxon signed-rank test results
│
├── run_significance_testing.py           # Statistical significance testing script
├── requirements.txt
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU with **≥16 GB VRAM** (≥8 GB with 4-bit NF4 fallback)
- Hugging Face access to [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### Setup

```bash
git clone https://github.com/MrArnav69/AnchorSum.git
cd AnchorSum

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_trf

huggingface-cli login
```

### External Evaluation Dependencies

The metric evaluation scripts depend on four external repositories. Clone each into **both** `scripts/Component_Ablation/` and `scripts/Revision_Depth/`:

```bash
for DIR in scripts/Component_Ablation scripts/Revision_Depth; do
  cd $DIR
  git clone https://github.com/yuh-zha/AlignScore
  git clone https://github.com/neulab/BARTScore
  git clone https://github.com/tingofurro/summac
  git clone https://github.com/maszhongming/UniEval
  cd ../..
done
```

Download the **AlignScore-large checkpoint** (~1.2 GB):

```bash
wget https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt \
     -O scripts/Component_Ablation/AlignScore/AlignScore-large.ckpt
cp scripts/Component_Ablation/AlignScore/AlignScore-large.ckpt \
   scripts/Revision_Depth/AlignScore/AlignScore-large.ckpt
```

---

## Quick Start

```python
from src.pipeline import anchorsumpipeline

pipeline = anchorsumpipeline(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    nli_model_name="cross-encoder/nli-deberta-v3-large",
    entity_model_name="en_core_web_trf",
    token="<your_hf_token>",
    max_revisions=1,    # T_max (recommended: 1)
    nli=True,           # Enable NLI audit
    entity=True,        # Enable Entity Guard
    revision=True       # Enable iterative revision
)

documents = "Article 1 text... ||| Article 2 text... ||| Article 3 text..."
result = pipeline.process(documents)

print(result["final_summary"])
print(f"Revisions performed: {result['revisions']}")
```

### Configuration Flags

| Flag | Type | Default | Effect |
|---|---|---|---|
| `max_revisions` | `int` | `1` | Maximum revision iterations T_max |
| `nli` | `bool` | `True` | Enable sentence-level NLI audit |
| `entity` | `bool` | `True` | Enable anchor extraction + hallucination filter |
| `revision` | `bool` | `True` | Enable flag-guided LLM revision |

> ⚠️ **`max_revisions=2` is not recommended.** A second revision cycle triggers verifier exploitation — inflating SummaCConv while collapsing BARTScore s→d on 100% of instances. See the paper (§8) for full analysis.

---

## Reproducing Results

### Step 1 — Generate Ablation Summaries

```bash
# Component ablation: base, no_nli, no_entity, full (T_max=1)
# Checkpoints every 50 samples; safe to interrupt and resume
python ablations/Component_Ablation/run_all_sequential.py

# Revision depth: full pipeline with T_max=2
python ablations/Revision_Depth/revision2.py
```

### Step 2 — Compute Evaluation Metrics

```bash
# Component Ablation
python scripts/Component_Ablation/evaluate_summac_final.py
python scripts/Component_Ablation/evaluate_alignscore_simple.py
python scripts/Component_Ablation/evaluate_bartscore_simple.py
python scripts/Component_Ablation/evaluate_rouge_bertscore_simple.py
python scripts/Component_Ablation/evaluate_bertscore_xlarge.py
python scripts/Component_Ablation/evaluate_unieval_fluency_simple.py

# Revision Depth
python scripts/Revision_Depth/evaluate_summac_full_revisions_2.py
python scripts/Revision_Depth/evaluate_bartscore_full_revisions_2.py
python scripts/Revision_Depth/evaluate_alignscore_full_revisions_2.py
python scripts/Revision_Depth/evaluate_rouge_bert_full_revisions_2.py
python scripts/Revision_Depth/evaluate_unieval_fluency_full_revisions_2.py
```

### Step 3 — Run Statistical Significance Tests

```bash
python run_significance_testing.py
# Outputs: Significance_Testing/wilcoxon_results.csv
```

### Reproducibility Parameters

| Parameter | Value |
|---|---|
| Random seed | `42` |
| Dataset | Multi-News `test` split ([`Awesome075/multi_news_parquet`](https://huggingface.co/datasets/Awesome075/multi_news_parquet)) |
| Sample size | 500 (498 after GPU memory exclusions) |
| Anchor budget N | 15 |
| Generator | `meta-llama/Llama-3.1-8B-Instruct` (8.03B params) |
| NLI model | `cross-encoder/nli-deberta-v3-large` (435M params) |
| NER model | `en_core_web_trf` (RoBERTa-base, 125M params) |
| Decoding | Greedy (`do_sample=False`), `repetition_penalty=1.1` |
| Max new tokens | 1024 |
| Precision | FP32 (4-bit NF4 fallback available) |
| NLI premise truncation | 512 subword tokens |

---

## Results Summary

All results below are reported over n=498 Multi-News test instances. Full analysis, statistical tests, and discussion are in the paper.

### Component Ablation — Primary Faithfulness Metrics

| Configuration | SummaCConv ↑ | AlignScore ↑ | BARTScore s→d ↑ | BARTScore d→s ↑ |
|:---|:---:|:---:|:---:|:---:|
| Base | 0.715 ± 0.034 | 0.828 ± 0.094 | −4.038 ± 0.383 | −3.470 ± 0.428 |
| no_entity | 0.712 ± 0.034 | 0.829 ± 0.094 | −4.045 ± 0.376 | −3.487 ± 0.426 |
| no_nli | 0.733 ± 0.041 | 0.824 ± 0.095 | −4.027 ± 0.392 | −3.481 ± 0.451 |
| **Full (AnchorSum)** | **0.733 ± 0.043** | 0.822 ± 0.093 | **−4.019 ± 0.394** | **−3.469 ± 0.450** |

### Component Ablation — Secondary Metrics

| Configuration | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F₁ | UniEval Fluency |
|:---|:---:|:---:|:---:|:---:|:---:|
| Base | **0.399** | 0.117 | **0.183** | **0.619** | 0.945 |
| no_entity | 0.396 | 0.113 | 0.181 | 0.617 | **0.949** |
| no_nli | 0.383 | **0.119** | 0.178 | 0.612 | 0.940 |
| **Full** | 0.384 | **0.119** | 0.178 | 0.611 | 0.944 |

### Revision Depth — Verifier Exploitation Signature

| Config | SummaCConv | AlignScore | BARTScore s→d | ROUGE-1 | BERTScore F₁ | UniEval |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Full (T_max=1) | 0.733 | **0.822** | **−4.019** | **0.384** | **0.611** | **0.944** |
| Full (T_max=2) | **0.918** | 0.821 | −6.585 | 0.384 | 0.610 | 0.943 |
| **Δ (T₂ − T₁)** | **+0.185** | −0.001 | **−2.566** | <0.001 | −0.001 | −0.001 |

> The +0.185 SummaCConv gain at T_max=2 is **verifier exploitation**, not genuine improvement. BARTScore s→d collapses by −2.566 nats on every single instance (W=0, p=2.68×10⁻⁸³), while AlignScore shows Δ<0.001. See paper §8 for the full mechanistic analysis and three-condition detection protocol.

### LLM-as-Judge Evaluation (n=50)

Dual-judge panel: DeepSeek-V3.1 (DeepThink) + Qwen3.5-Plus (thinking mode). G-Eval framework with anonymized model identities.

| System | Faithfulness | Coverage | Consistency | Fluency | **Jury** |
|:---|:---:|:---:|:---:|:---:|:---:|
| BART-large-CNN | 6.43 | 6.10 | 6.57 | 6.75 | 6.46 |
| PRIMERA-MultiNews† | 6.68 | 6.40 | 6.86 | 7.13 | 6.77 |
| PEGASUS-MultiNews | 7.37 | 7.11 | 7.52 | 7.57 | 7.39 |
| Base LLaMA | 7.99 | 7.88 | 8.20 | 8.25 | 8.08 |
| **AnchorSum (T=1)** | **8.22** | **8.04** | 8.42 | 8.51 | **8.30** |
| AnchorSum (T=2) | 8.20 | 8.10 | **8.41** | **8.48** | **8.30** |

<sub>† PRIMERA-MultiNews evaluated on raw concatenated documents without entity-pyramid preprocessing; scores represent a lower bound.</sub>

Full judge transcripts: [DeepSeek-V3.1](https://chat.deepseek.com/share/thkcnnbqyclko2elhx) · [Qwen3.5-Plus](https://chat.qwen.ai/s/3badcaee-65f9-4008-bc33-66e4d6c820eb)

### Statistical Significance (Wilcoxon Signed-Rank Tests, n=498)

| Test | W | p-value | Survives Bonferroni |
|:---|:---:|:---:|:---:|
| Full vs Base — SummaCConv | 26,687 | 4.49×10⁻²⁸ | ✓ |
| Full vs Base — BARTScore s→d | 50,753 | 4.01×10⁻⁴ | ✓ |
| T₂ vs T₁ — SummaCConv | 894 | 5.68×10⁻⁸¹ | ✓ |
| T₂ vs T₁ — BARTScore s→d | 0 | 2.68×10⁻⁸³ | ✓ |
| no_entity vs Base — SummaCConv | 43,895 | 2.91×10⁻⁵ | ✓ |

---

## Evaluation Metrics

| Metric | Backbone | What It Measures |
|---|---|---|
| **SummaCConv** | ViT-C NLI fine-tune | Sentence-level entailment consistency |
| **AlignScore** | RoBERTa-Large (7 NLU tasks) | Cross-framework factual alignment |
| **BARTScore s→d** | BART-large-CNN | log P(summary \| source) — register-sensitive |
| **BARTScore d→s** | BART-large-CNN | log P(source \| summary) — coverage |
| **ROUGE-1/2/L** | n-gram overlap | Reference lexical overlap |
| **BERTScore F₁** | DeBERTa-XLarge-MNLI | Token-level semantic similarity |
| **UniEval Fluency** | T5-large evaluator | Linguistic fluency ∈ [0,1] |

---

## Module Reference

| Module | Description |
|---|---|
| `src/pipeline.py` | Orchestrates Phases 0–3. The `nli`, `entity`, and `revision` flags enable all four ablation configurations from a single entrypoint. |
| `src/llm_summarizer.py` | Wraps Llama-3.1-8B-Instruct with structured generation and revision prompts. Greedy decoding with `repetition_penalty=1.1`. |
| `src/verification/nli_verifier.py` | Sentence-level NLI audit using `cross-encoder/nli-deberta-v3-large`. Tokenizes summary into sentences (NLTK Punkt), classifies each against source truncated to 512 tokens. |
| `src/verification/entity_guard.py` | Anchor extraction via `en_core_web_trf` + conjunctive hallucination filter requiring absence from both NER output and raw source string. |
| `ablations/ablation_base_runner.py` | Shared experiment infrastructure: dataset loading, per-50-sample checkpointing, graceful interrupt-resume. |

---

## Dataset

| Property | Value |
|---|---|
| Corpus | [Multi-News](https://huggingface.co/datasets/Awesome075/multi_news_parquet) (Fabbri et al., 2019) |
| Split | `test` |
| Sample size | 500 (498 after GPU memory exclusions) |
| Sampling | `Dataset.shuffle(seed=42).select(range(500))` |
| Mean documents per instance | 2.8 |
| Mean source length | 1,247 tokens |
| Cached subset | `data/multi_news_500_samples.json` |

---

## Computational Requirements

| Stage | Per-sample time | Total (498 instances) |
|---|---|---|
| Base generation only | ~37s | ~5.1 GPU-hours |
| Full pipeline (T_max=1) | ~102s | ~14.2 GPU-hours |
| Full pipeline (T_max=2) | ~186s | ~25.8 GPU-hours |

<sub>Benchmarked on a single NVIDIA A40 (45 GB HBM2).</sub>

---

## AI Use Disclosure

This project utilized AI assistants (including Anthropic Claude, Google Gemini, Kimi, and other AI tools) for code assistance, debugging, and help with evaluation of metrics. All core research contributions — algorithmic design, experimental methodology, results, and conclusions — are the original work of the author.



