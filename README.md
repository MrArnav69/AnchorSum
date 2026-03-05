<p align="center">
  <h1 align="center">AnchorSum</h1>
  <p align="center">
    <strong>Improving Multi-Document Summarization Faithfulness<br/>through NLI and Entity-Gated Iterative Refinement</strong>
  </p>
  
  <p align="center">
    <a href="#overview">Overview</a> ·
    <a href="#key-contributions">Contributions</a> ·
    <a href="#architecture">Architecture</a> ·
    <a href="#repository-structure">Repository</a> ·
    <a href="#installation">Installation</a> ·
    <a href="#usage">Usage</a> ·
    <a href="#reproducing-paper-results">Reproduce</a> ·
    <a href="#results">Results</a> ·
  </p>
</p>

---

## Overview

Abstractive multi-document summarization systems optimized for conditional log-likelihood routinely **hallucinate** — generating entities, events, and statistics with no grounding in any source document. **AnchorSum** addresses this without any task-specific fine-tuning through a four-phase modular pipeline:

1. **Anchor extraction** — a RoBERTa-base NER pipeline ranks named entities by corpus frequency, producing a constraint set $\mathcal{A}$ that simultaneously guides generation and provides post-hoc verification targets.
2. **Anchor-conditioned draft generation** — `Meta-Llama-3.1-8B-Instruct` generates an initial draft under greedy decoding with $\mathcal{A}$ injected as a soft semantic constraint.
3. **Dual-mode faithfulness auditing** — two independent, complementary verifiers assess the draft: a cross-encoder NLI model (`cross-encoder/nli-deberta-v3-large`) checks sentence-level entailment; a conjunctive entity filter checks anchor coverage and extrinsic hallucination. Together they produce a structured flag set $\mathcal{F} = \mathcal{F}_\text{NLI} \cup \mathcal{F}_\text{ENT}$.
4. **Flag-guided iterative revision** — the LLM is re-prompted in an expert-editor role with $\mathcal{F}$ as a numbered corrective list. Critically, feedback originates from **external neural verifiers**, not LLM self-assessment, eliminating sycophantic self-evaluation artifacts.

> **Evaluated on 500 Multi-News test instances** across six automatic metrics and on 50 instances via a dual-judge LLM-as-judge panel (DeepSeek-R1 + Qwen3.5-Plus thinking mode). AnchorSum achieves a **6.3% relative SummaCConv improvement** over the unaugmented generator and outperforms PEGASUS-MultiNews, PRIMERA-MultiNews, and BART-large-CNN across every faithfulness dimension.

---

## Key Contributions

| # | Contribution | Location |
|---|---|---|
| 1 | **Anchor-conditioned generation** — corpus-frequency entity anchors serve as both generative soft constraints and post-hoc verification targets | `src/verification/entity_guard.py` |
| 2 | **Dual-mode auditing** — NLI (sentence-level semantic inconsistency) and Entity Guard (extrinsic hallucination) address non-overlapping failure modes in Maynez et al.'s hallucination taxonomy | `src/verification/` |
| 3 | **Externally grounded revision** — structured flag feedback from independent neural verifiers, explicitly avoiding Self-Refine-style sycophancy | `src/llm_summarizer.py` |
| 4 | **Verifier exploitation as a novel failure mode** — a second revision cycle inflates SummaCConv by +0.185 while collapsing BARTScore s→d by −2.566 nats at <0.4% output length change; we provide a principled multi-metric triangulation methodology for detecting it | `ablations/Revision_Depth/` |

---

## Architecture

```
Source Corpus D = d₁ ‖ … ‖ dₖ
         │
         ▼
┌─────────────────────────────────────┐
│  PHASE 0 — Anchor Extraction        │
│  spaCy en_core_web_trf (125M)        │
│  Top-N entities by corpus frequency  │
│  A = {a₁, …, aₙ}   (N = 15)        │
└──────────────┬──────────────────────┘
               │ A
               ▼
┌─────────────────────────────────────┐
│  PHASE 1 — Draft Generation         │
│  Meta-Llama-3.1-8B-Instruct         │
│  Greedy decoding · rep. penalty 1.1 │
│  A injected as soft constraint       │
│  Ŝ⁽⁰⁾ ← LLM(D, A)                 │
└──────────────┬──────────────────────┘
               │ Ŝ⁽ⁱ⁾
               ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 2 — Dual-Mode Faithfulness Audit                 │
│                                                         │
│  ┌───────────────────────────┐  ┌─────────────────────┐ │
│  │  2a: NLI Audit            │  │  2b: Entity Guard   │ │
│  │  cross-encoder/            │  │  Coverage check:    │ │
│  │  nli-deberta-v3-large     │  │  M = anchors absent │ │
│  │  435M · SNLI + MultiNLI   │  │  from Ŝ             │ │
│  │                           │  │                     │ │
│  │  Each sentence Ŝᵢ vs D   │  │  Hallucination:     │ │
│  │  → ENTAILMENT / NEUTRAL   │  │  H = entities in Ŝ  │ │
│  │    / CONTRADICTION        │  │  absent from D      │ │
│  │                           │  │  (conjunctive filter│ │
│  │  F_NLI = {non-entailed}   │  │  suppresses FP)     │ │
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
                               │  LLM.Revise(D, Ŝ⁽ⁱ⁾, F)│
                               │  Expert-editor role    │
                               │  Flags as numbered list│
                               │  Ŝ⁽ⁱ⁺¹⁾ → Phase 2     │
                               └────────────────────────┘

  Convergence: i* = min{i | F⁽ⁱ⁾ = ∅  or  i = T_max}
  Default: T_max = 1
```

---

## Repository Structure

```
AnchorSum/
│
├── src/                                  # Core pipeline source
│   ├── __init__.py
│   ├── pipeline.py                       # Main orchestrator (Phase 0–3 composition)
│   ├── llm_summarizer.py                 # LLaMA draft generation + revision prompting
│   └── verification/
│       ├── __init__.py
│       ├── nli_verifier.py               # DeBERTa-v3-Large sentence-level NLI audit
│       └── entity_guard.py               # spaCy anchor extraction + hallucination filter
│
├── ablations/                            # Experiment runners
│   ├── ablation_base_runner.py           # Shared base: dataset loading, checkpointing
│   ├── Component_Ablation/
│   │   └── run_all_sequential.py         # 4 configs × 500 samples (sequential)
│   └── Revision_Depth/
│       └── revision2.py                  # Full pipeline, T_max = 2
│
├── scripts/                              # Per-metric evaluation scripts
│   ├── Component_Ablation/
│   │   ├── evaluate_rouge_bertscore_simple.py      # ROUGE-1/2/L (BERTScore Broken - Do Not Use)
│   │   ├── evaluate_bertscore_xlarge.py            # BERTScore (DeBERTa-XLarge-MNLI)
│   │   ├── evaluate_alignscore_simple.py           # AlignScore (RoBERTa-Large, NLI-SP)
│   │   ├── evaluate_bartscore_simple.py            # BARTScore (BART-Large-CNN, s→d & d→s)
│   │   ├── evaluate_summac_final.py                # SummaCConv (ViT-C backbone)
│   │   └── evaluate_unieval_fluency_simple.py      # UniEval fluency dimension
│   └── Revision_Depth/
│       ├── evaluate_rouge_bert_full_revisions_2.py
│       ├── evaluate_bartscore_full_revisions_2.py
│       ├── evaluate_alignscore_full_revisions_2.py
│       ├── evaluate_summac_full_revisions_2.py
│       └── evaluate_unieval_fluency_full_revisions_2.py
│
├── data/
│   ├── multi_news_500_samples.json       # Cached 500-sample subset (seed=42)
│   ├── document.json                     # Sample outputs with multi-model comparisons
│   └── ablations/                        # Generated summaries per configuration
│       ├── base/                         # No auditing, no revision
│       ├── no_nli/                       # Entity Guard only
│       ├── no_entity/                    # NLI Audit only
│       ├── full/                         # Full pipeline (T_max = 1)
│       └── full_revisions_2/             # Full pipeline (T_max = 2)
│
├── Results/                              # Computed metric outputs
│   ├── Component Ablation/
│   │   ├── rouge_bert_simple_results/ #BERTScore Broken - Do Not Use (Use bertscore_xlarge_results.py)
│   │   ├── bertscore_xlarge_results/
│   │   ├── alignscore_results/
│   │   ├── bartscore_simple_results/
│   │   ├── summac_final_results/
│   │   └── unieval_fluency_results/
│   └── Revision Depth/
│       ├── rouge_bert_full_revisions_2_results/
│       ├── summac_full_revisions_2_results/
│       ├── alignscore_full_revisions_2_results/
│       ├── bartscore_full_revisions_2_results/
│       └── unieval_fluency_full_revisions_2_results/
│
├── requirements.txt
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU with **≥16 GB VRAM** (for LLaMA 3.1-8B at FP32; ≥8 GB with 4-bit NF4 fallback)
- Hugging Face account with access to [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/MrArnav69/AnchorSum.git
cd AnchorSum

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download the spaCy transformer model
python -m spacy download en_core_web_trf

# 5. Authenticate with Hugging Face
huggingface-cli login
# or: export HF_TOKEN=your_token_here
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

Download the **AlignScore-large checkpoint** (~1.2 GB) into each `AlignScore/` directory:

```bash
wget https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt \
     -O scripts/Component_Ablation/AlignScore/AlignScore-large.ckpt
cp scripts/Component_Ablation/AlignScore/AlignScore-large.ckpt \
   scripts/Revision_Depth/AlignScore/AlignScore-large.ckpt
```

---

## Usage

### Programmatic API

```python
from src.pipeline import anchorsumpipeline

pipeline = anchorsumpipeline(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    nli_model_name="cross-encoder/nli-deberta-v3-large",
    entity_model_name="en_core_web_trf",
    token="<your_hf_token>",
    max_revisions=1,    # T_max; default and recommended (see warning below)
    nli=True,           # Enable NLI audit (Phase 2a)
    entity=True,        # Enable Entity Guard (Phase 2b)
    revision=True       # Enable iterative revision (Phase 3)
)

documents = "Article 1: ... ||| Article 2: ... ||| Article 3: ..."
result = pipeline.process(documents)

print(result["final_summary"])
print(f"Revisions performed: {result['revisions']}")
# result["history"] contains (draft, flags) pairs per iteration
```

### Configuration Flags

| Flag | Type | Default | Effect |
|------|------|---------|--------|
| `max_revisions` | `int` | `1` | Maximum revision iterations $T_\text{max}$ |
| `nli` | `bool` | `True` | Enable sentence-level NLI audit |
| `entity` | `bool` | `True` | Enable anchor extraction + hallucination filter |
| `revision` | `bool` | `True` | Enable flag-guided LLM revision |

> ⚠️ **`max_revisions=2` is not recommended for production use.** See the [Verifier Exploitation](#%EF%B8%8F-verifier-exploitation-warning--revision-depth-analysis-n--500) section in Results for a full explanation.

---

## Reproducing Paper Results

### Step 1 — Generate Ablation Summaries

```bash
# Component ablation: runs base, no_nli, no_entity, and full sequentially
# Checkpoints every 50 samples; safe to interrupt and resume
python ablations/Component_Ablation/run_all_sequential.py

# Revision depth analysis: full pipeline with T_max = 2
python ablations/Revision_Depth/revision2.py
```

### Step 2 — Compute Evaluation Metrics

```bash
# ── Component Ablation ──────────────────────────────────────────
python scripts/Component_Ablation/evaluate_summac_final.py
python scripts/Component_Ablation/evaluate_alignscore_simple.py
python scripts/Component_Ablation/evaluate_bartscore_simple.py
python scripts/Component_Ablation/evaluate_rouge_bertscore_simple.py
python scripts/Component_Ablation/evaluate_bertscore_xlarge.py
python scripts/Component_Ablation/evaluate_unieval_fluency_simple.py

# ── Revision Depth ──────────────────────────────────────────────
python scripts/Revision_Depth/evaluate_summac_full_revisions_2.py
python scripts/Revision_Depth/evaluate_bartscore_full_revisions_2.py
python scripts/Revision_Depth/evaluate_alignscore_full_revisions_2.py
python scripts/Revision_Depth/evaluate_rouge_bert_full_revisions_2.py
python scripts/Revision_Depth/evaluate_unieval_fluency_full_revisions_2.py
```

### Reproducibility Parameters

| Parameter | Value |
|-----------|-------|
| Random seed | `42` |
| Dataset split | `test` (Multi-News) |
| Sample size | `500` |
| Anchor budget $N$ | `15` |
| Generator | `meta-llama/Llama-3.1-8B-Instruct` |
| NLI model | `cross-encoder/nli-deberta-v3-large` |
| NER model | `en_core_web_trf` |
| Decoding | Greedy (`do_sample=False`), `repetition_penalty=1.1` |
| Max new tokens | `1024` |
| Precision | `float32` (4-bit NF4 fallback available) |
| NLI premise truncation | `512` subword tokens |

---

## Results

### Component Ablation — Faithfulness and Semantic Fidelity (n = 500)

| Configuration | SummaCConv ↑ | AlignScore ↑ | BARTScore s→d ↓ | BARTScore d→s ↓ |
|:---|:---:|:---:|:---:|:---:|
| Base | 0.715 ± 0.034 | 0.828 ± 0.094 | −4.038 ± 0.383 | −3.470 ± 0.428 |
| no\_entity †| 0.712 ± 0.034 | 0.829 ± 0.094 | −4.045 ± 0.376 | −3.487 ± 0.426 |
| no\_nli | 0.733 ± 0.041 | 0.824 ± 0.095 | −4.027 ± 0.392 | −3.481 ± 0.451 |
| **Full (AnchorSum)** | **0.733 ± 0.043** | 0.822 ± 0.093 | **−4.019 ± 0.394** | **−3.469 ± 0.450** |

> † **no\_entity degrades below base** (SummaCConv 0.712 vs 0.715), establishing the Entity Guard as a *necessary condition* for faithfulness improvement — its removal actively worsens faithfulness rather than merely failing to improve it.

### Component Ablation — Lexical, Neural, and Fluency Metrics (n = 500)

| Configuration | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F₁ | UniEval Fluency |
|:---|:---:|:---:|:---:|:---:|:---:|
| Base | **0.399** | 0.117 | **0.183** | **0.619** | 0.945 |
| no\_entity | 0.396 | 0.113 | 0.181 | 0.617 | **0.949** |
| no\_nli | 0.383 | **0.119** | 0.178 | 0.612 | 0.940 |
| **Full** | 0.384 | **0.119** | 0.178 | 0.611 | 0.944 |

> Modest ROUGE-1/L declines for Full vs. Base reflect the faithfulness–reference-lexicality tension: revisions correcting factual errors are not lexically closer to references written without revision signals.

---

### ⚠️ Verifier Exploitation Warning — Revision Depth Analysis (n = 500)

This is the paper's most important negative finding. The SummaCConv gain at `T_max=2` is **not** a quality improvement — it is a documented failure mode.

| Config | SummaCConv | AlignScore | BARTScore s→d | ROUGE-1 | BERTScore F₁ | UniEval |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Full (T_max=1) | 0.733 | **0.822** | **−4.019** | **0.384** | **0.611** | **0.944** |
| Full (T_max=2) | **0.918** | 0.821 | −6.585 | 0.384 | 0.610 | 0.943 |
| Δ (T₂ − T₁) | **+0.185** | −0.001 | **−2.566** | <0.001 | −0.001 | −0.001 |

**What is happening:** A second revision cycle inflates SummaCConv by +0.185 while BARTScore s→d collapses from −4.019 to −6.585 (**−2.566 nats, −64%**) at an output length change of only +10 characters (<0.4%). This is **verifier exploitation**: the model rewrites into propositionally atomic declarative sentences ("Subject did X.") that maximize per-sentence NLI entailment scores, abandoning the discourse structure of fluent abstractive prose. The auditor is satisfied; the summary quality is not improved.

AlignScore (not NLI-based) remaining at Δ = −0.001 is the decisive diagnostic: genuine faithfulness gains transfer across metric frameworks; NLI-optimized register shifts do not. This is a concrete instance of **Goodhart's Law in iterative NLG refinement**.

**`T_max=1` is the principled default and the recommended production setting.**

---

### LLM-as-Judge Comparative Evaluation (n = 50)

Evaluated against PEGASUS-MultiNews, PRIMERA-MultiNews, and BART-large-CNN using a dual-judge panel (DeepSeek-R1 + Qwen3.5-Plus in thinking mode) following the G-Eval framework. Model identities were anonymized. Jury = unweighted mean of both judges.

| System | Faithfulness | Coverage | Consistency | Fluency | **Overall (Jury)** |
|:---|:---:|:---:|:---:|:---:|:---:|
| BART-large-CNN | 6.43 | 6.10 | 6.57 | 6.75 | 6.46 |
| PRIMERA-MultiNews | 6.68 | 6.40 | 6.86 | 7.13 | 6.77 |
| PEGASUS-MultiNews | 7.37 | 7.11 | 7.52 | 7.57 | 7.39 |
| Base LLaMA (no auditing) | 7.99 | 7.88 | 8.20 | 8.25 | 8.08 |
| **AnchorSum (T_max=1)** | **8.22** | **8.04** | 8.42 | 8.51 | **8.30** |
| AnchorSum (T_max=2) | 8.20 | 8.10 | **8.41** | **8.48** | **8.30** |

Both AnchorSum configurations outperform all external baselines across every dimension. The jury ties T_max=1 and T_max=2 at 8.30 — consistent with the automatic metrics showing T_max=2 is holistically plausible to fluency-sensitive judges while BARTScore exposes the latent register degradation.

Full per-model judge reports: [DeepSeek-R1](https://chat.deepseek.com/share/thkcnnbqyclko2elhx) · [Qwen3.5-Plus](https://chat.qwen.ai/s/3badcaee-65f9-4008-bc33-66e4d6c820eb)

---

## Evaluation Metric Reference

| Metric | Backbone | Dimension | Key Role in This Paper |
|--------|----------|-----------|----------------------|
| **SummaCConv** | ViT-C NLI fine-tune | Sentence entailment consistency | Primary faithfulness metric; also the exploited metric at T_max=2 |
| **AlignScore** | RoBERTa-Large (7 NLU tasks) | Cross-framework factual alignment | Non-NLI cross-check; decisive for verifier exploitation diagnosis |
| **BARTScore s→d** | BART-large-CNN | log P(summary \| source) | Register-sensitive; detects atomic-declarative rewriting |
| **BARTScore d→s** | BART-large-CNN | log P(source \| summary) | Source coverage under generative model |
| **ROUGE-1/2/L** | n-gram overlap | Reference lexical overlap | Benchmark standard; modest declines are expected and non-indicative |
| **BERTScore F₁** | DeBERTa-XLarge-MNLI | Token-level semantic similarity | Precision/recall split isolates anchor-driven length effects |
| **UniEval Fluency** | T5-large evaluator | Linguistic fluency ∈ [0,1] | Monitors fluency degradation across revision iterations |

---

## Module Reference

### `src/pipeline.py` — Pipeline Orchestrator
Composes Phases 0–3 into the generate → verify → revise loop. The `nli`, `entity`, and `revision` flags enable all four component ablation configurations from a single entrypoint.

### `src/llm_summarizer.py` — LLM Generation and Revision
Wraps `Meta-Llama-3.1-8B-Instruct` with two structured prompts: a journalist-role generation prompt injecting $\mathcal{D}$ and $\mathcal{A}$, and an expert-editor revision prompt injecting $\mathcal{D}$, $\hat{S}^{(i)}$, and $\mathcal{F}$ as a numbered corrective list. Both use greedy decoding with `repetition_penalty=1.1`.

### `src/verification/nli_verifier.py` — NLI Faithfulness Audit
Tokenizes the summary into sentences (NLTK punkt), pairs each against the source corpus truncated to 512 subword tokens, and classifies with `cross-encoder/nli-deberta-v3-large` (435M params, SNLI + MultiNLI fine-tuned). Returns $\mathcal{F}_\text{NLI}$ — the set of non-entailed sentences with their predicted labels.

### `src/verification/entity_guard.py` — Entity Guard
Two functions: **(1) Anchor extraction** — runs `en_core_web_trf` over $\mathcal{D}$, filters to {PERSON, ORG, GPE, LOC, DATE, MONEY, PERCENT}, discards short spans (|e| ≤ 2), ranks by corpus frequency, returns Top-N. **(2) Hallucination detection** — applies a conjunctive filter requiring a candidate entity to be absent from *both* the NER output of $\mathcal{D}$ *and* the lowercased raw string of $\mathcal{D}$ before flagging, conservatively suppressing false positives from NER pipeline gaps.

### `ablations/ablation_base_runner.py` — Shared Experiment Runner
Handles dataset loading from `data/multi_news_500_samples.json`, per-sample checkpointing (every 50 samples), result serialization to `data/ablations/`, and graceful interrupt-resume. All ablation runners inherit from this base.

---

## Dataset

| Property | Value |
|----------|-------|
| Corpus | [Multi-News](https://huggingface.co/datasets/Awesome075/multi_news_parquet) (Fabbri et al., 2019) |
| Split | `test` |
| Sample size | 500 (deterministic shuffle, `seed=42`) |
| Sampling | `Dataset.shuffle(seed=42).select(range(500))` |
| Excluded instances | 2 (GPU memory exceeded at evaluation batch size) |
| Cached subset | `data/multi_news_500_samples.json` |

---


## AI Use Disclosure

This project utilized AI assistants (including Anthropic Claude, Google Gemini, Kimi, and other AI tools) for Code Assistance, Debugging, and help with Evaluation of Metrics. 

All core research contributions, algorithmic designs (NLI-Entity Gated Refinement), and final evaluation results are the original work of the author.
