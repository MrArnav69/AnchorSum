<p align="center">
  <h1 align="center">AnchorSum</h1>
  <p align="center">
    <strong>Supplementary Material & Code Repository</strong><br/>
    <em>Verifier Exploitation in NLI-Guided Iterative Refinement: A Controlled Empirical Analysis</em>
  </p>
  <p align="center">
    <em>Under review as submission to Transactions on Machine Learning Research (TMLR)</em>
  </p>
  <p align="center">
    <a href="#overview">Overview</a> ·
    <a href="#system-architecture">Architecture</a> ·
    <a href="#repository-structure">Repository Structure</a> ·
    <a href="#file-level-documentation">File Documentation</a> ·
    <a href="#installation">Installation</a> ·
    <a href="#reproducing-results">Reproducing Results</a> ·
    <a href="#results-summary">Results</a> ·
    <a href="#evaluation-metrics">Metrics</a>
  </p>
</p>

---

## Overview

This repository contains the complete source code, experimental infrastructure, precomputed results, and evaluation scripts accompanying the paper *"Verifier Exploitation in NLI-Guided Iterative Refinement: A Controlled Empirical Analysis."*

**AnchorSum** is a modular, training-free pipeline for faithful multi-document summarization. It combines entity-guided anchor extraction, anchor-conditioned draft generation, dual-mode faithfulness auditing (sentence-level NLI + entity hallucination filtering), and flag-guided iterative revision — all without modifying any model weights. The pipeline serves as the controlled experimental vehicle for demonstrating that **verifier exploitation** — satisfying an NLI auditing metric while degrading real faithfulness — manifests even in zero-gradient, prompt-only refinement loops.

### Principal Findings

1. **Faithful single-cycle refinement.** AnchorSum at T<sub>max</sub>=1 achieves a **6.3% relative SummaCConv inconsistency reduction** over its unaugmented base (p = 4.49 × 10⁻²⁸, Wilcoxon signed-rank, Bonferroni-corrected).
2. **Verifier exploitation at T<sub>max</sub>=2.** A second revision cycle inflates SummaCConv by +0.185 while collapsing BARTScore s→d by −2.566 nats — **Wilcoxon W = 0 across all 498 instances** — confirming that the gain is verifier exploitation, not genuine faithfulness improvement. AlignScore registers Δ < 0.001, establishing that exploited NLI scores do not transfer to independent faithfulness frameworks.
3. **Superiority over fine-tuned baselines.** AnchorSum outperforms PEGASUS-MultiNews, PRIMERA-MultiNews, and BART-large-CNN across all dimensions in a dual-judge LLM-as-judge evaluation (jury score: 8.30/10).

### Three-Condition Detection Protocol

The paper formalizes an annotation-free diagnostic for detecting verifier exploitation in any NLI-guided iterative refinement pipeline. The three necessary conditions are:

| Condition | Criterion | Observed Value |
|:---|:---|:---|
| (i) Large NLI-metric gain | ΔSummaCConv > 0.1 | +0.185 |
| (ii) No independent faithfulness transfer | ΔAlignScore ≤ 0 | < 0.001 |
| (iii) Generative log-probability collapse | ΔBARTScore<sub>s→d</sub> < −1.0 on ≥ 90% of instances | −2.566 (100%, W = 0) |

All three conditions are necessary; no proper subset is sufficient. The protocol requires no human annotation, no model retraining, and no access to model internals.

---

## System Architecture

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

The pipeline decouples generation from verification: the LLM generates and revises, while independent neural auditors (DeBERTa-v3-Large for NLI, spaCy RoBERTa-base for entity grounding) supply structured feedback. This separation enables precise mechanistic attribution in the component ablation (§6 of the paper).

---

## Repository Structure

```
AnchorSum/
│
├── src/                                  # Core pipeline source code
│   ├── __init__.py                       #   Package initializer
│   ├── pipeline.py                       #   Main orchestrator — composes Phases 0–3
│   ├── llm_summarizer.py                 #   LLaMA draft generation + revision prompting
│   └── verification/
│       ├── __init__.py                   #   Subpackage initializer
│       ├── nli_verifier.py               #   DeBERTa-v3-Large sentence-level NLI audit
│       └── entity_guard.py               #   spaCy anchor extraction + hallucination filter
│
├── ablations/                            # Experiment runners for all ablation configurations
│   ├── ablation_base_runner.py           #   Shared base: dataset loading, checkpointing, experiment loop
│   ├── Component_Ablation/
│   │   └── run_all_sequential.py         #   Runs all 4 component configurations × 500 samples
│   └── Revision_Depth/
│       └── revision2.py                  #   Full pipeline at T_max=2 × 500 samples
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
├── Results/                              # Precomputed evaluation metric outputs (CSV)
│   ├── Component Ablation/
│   │   ├── summac_final_results/         #   Per-instance + summary CSVs (4 configs)
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
├── notebooks/                            # (Reserved for exploratory analysis)
├── run_significance_testing.py           # Statistical significance testing script
├── requirements.txt                      # Python dependencies
└── .gitignore
```

---

## File-Level Documentation

This section provides a detailed description of every file in the repository, its role in the experimental pipeline, and instructions for its use.

### Core Pipeline (`src/`)

The `src/` package contains the complete AnchorSum pipeline implementation. All four ablation configurations (base, no_nli, no_entity, full) are controlled through constructor flags on a single entrypoint class — no code duplication is required.

#### `src/pipeline.py` — Pipeline Orchestrator

**Role:** Central orchestrator that composes Phases 0–3 into a single callable pipeline.

**Key class:** `anchorsumpipeline`

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `model_name` | `str` | `meta-llama/Llama-3.1-8B-Instruct` | Hugging Face identifier for the generative backbone |
| `nli_model_name` | `str` | `cross-encoder/nli-deberta-v3-large` | NLI cross-encoder model |
| `entity_model_name` | `str` | `en_core_web_trf` | spaCy transformer NER model |
| `token` | `str` | `None` | Hugging Face authentication token |
| `max_revisions` | `int` | `1` | Maximum revision iterations T<sub>max</sub> |
| `nli` | `bool` | `True` | Enable sentence-level NLI audit (Phase 2a) |
| `entity` | `bool` | `True` | Enable anchor extraction + hallucination filter (Phase 2b) |
| `revision` | `bool` | `True` | Enable flag-guided LLM revision (Phase 3) |

**Method:** `process(document, reference_summary=None) → dict`
- Executes the full pipeline on a `|||`-delimited multi-document input string.
- Returns a dictionary containing `initial_draft`, `final_summary`, `reference`, `history` (list of per-revision records with flags), and `num_revisions`.

**Usage:**
```python
from src.pipeline import anchorsumpipeline

pipeline = anchorsumpipeline(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    nli_model_name="cross-encoder/nli-deberta-v3-large",
    entity_model_name="en_core_web_trf",
    token="<your_hf_token>",
    max_revisions=1,
    nli=True,
    entity=True,
    revision=True
)

documents = "Article 1 text... ||| Article 2 text... ||| Article 3 text..."
result = pipeline.process(documents)
print(result["final_summary"])
print(f"Revisions performed: {result['num_revisions']}")
```

> ⚠️ **`max_revisions=2` is not recommended.** A second revision cycle triggers verifier exploitation — inflating SummaCConv while collapsing BARTScore s→d on 100% of instances. See the paper (§5) for the full mechanistic analysis.

---

#### `src/llm_summarizer.py` — Generative Backbone

**Role:** Wraps `meta-llama/Llama-3.1-8B-Instruct` (8.03B parameters, Grouped-Query Attention) for both initial draft generation (Phase 1) and flag-guided revision (Phase 3).

**Key class:** `singlepassmds`

**Design decisions:**
- **Greedy decoding** (`do_sample=False`) with `repetition_penalty=1.1` ensures deterministic, reproducible outputs.
- **FP32 precision** is enforced by default; a 4-bit NF4 quantization fallback is available via `load_in_4bit=True` for systems with < 16 GB VRAM.
- **Chat template application**: Uses the model's native chat template (via `apply_chat_template`) for proper instruction formatting.
- **Max new tokens**: 1024 (configurable).

**Prompt structure:**
- *Draft prompt (Phase 1):* System role as "expert investigative journalist." Anchors from Phase 0 are injected as mandatory inclusion constraints. Explicit instructions to avoid hallucination and to begin the summary immediately without preamble.
- *Revision prompt (Phase 3):* System role as "expert editor." Receives the current draft and a numbered list of audit flags. Explicit instructions to produce a complete revised summary fixing all flagged issues.

**Methods:**
| Method | Input | Output |
|:---|:---|:---|
| `generate_draft(documents, anchors, max_new_tokens)` | Source text + anchor list | Initial summary draft |
| `revise_draft(documents, draft, flags, max_new_tokens)` | Source text + current draft + flag list | Revised summary |

---

#### `src/verification/nli_verifier.py` — Sentence-Level NLI Audit (Phase 2a)

**Role:** Performs sentence-level natural language inference between the source corpus and each summary sentence, using `cross-encoder/nli-deberta-v3-large` (435M parameters, trained on SNLI + MultiNLI).

**Key class:** `nliverifier`

**Mechanism:**
1. Tokenizes the summary into sentences using NLTK Punkt.
2. For each sentence, constructs a (premise, hypothesis) pair where premise = source text and hypothesis = summary sentence.
3. Classifies the pair into {ENTAILMENT, CONTRADICTION, NEUTRAL} using the DeBERTa-v3-Large cross-encoder.
4. Non-entailed sentences (CONTRADICTION or NEUTRAL) are flagged as `F_NLI`.

**Critical architectural constraint:** The cross-encoder truncates the premise (source text) to **512 subword tokens**. This creates a positional bias that is the proximate cause of verifier exploitation: content grounded in source material beyond the 512-token window cannot be verified and is systematically removed during revision. See §5 of the paper for the full mechanistic analysis.

**Method:** `verify_draft(source_text, draft_summary) → (passed: List[str], flagged: List[str])`

---

#### `src/verification/entity_guard.py` — Entity-Level Grounding Audit (Phase 2b)

**Role:** Performs two entity-level faithfulness checks: anchor coverage verification and extrinsic hallucination detection.

**Key class:** `entityguard`

**Mechanism:**
1. **Anchor extraction** (`extract_anchors`): Extracts named entities from the source corpus using `en_core_web_trf` (RoBERTa-base, 125M parameters, OntoNotes 5.0). Retains entities with labels in {PERSON, ORG, GPE, LOC, DATE, MONEY, PERCENT}. Ranks by corpus frequency and selects the top-15 as anchors.
2. **Coverage check**: Verifies that all extracted anchors appear in the generated summary. Missing anchors are flagged as `F_cov`.
3. **Hallucination filter**: Identifies entities in the summary that are absent from both the NER output and the raw source string. Uses a conjunctive criterion — an entity is flagged only if it fails both checks — to minimize false positives. Flagged as `F_hal`.
4. The union `F_ENT = F_cov ∪ F_hal` is passed to Phase 3.

**Methods:**
| Method | Input | Output |
|:---|:---|:---|
| `extract_anchors(source_text)` | Source document string | Top-N anchor list |
| `verify_draft(source_anchors, source_text, draft_summary)` | Anchors + source + draft | List of entity-level flags |

---

### Experiment Runners (`ablations/`)

The `ablations/` directory contains the infrastructure for executing all experimental configurations reported in the paper.

#### `ablations/ablation_base_runner.py` — Shared Experiment Infrastructure

**Role:** Provides the `run_experiment()` function used by all experiment scripts. Handles dataset loading, seeded sampling, pipeline initialization with per-configuration flags, per-50-sample checkpointing, and graceful interrupt-resume.

**Key function:** `run_experiment(config_name, ablation_flags, max_revisions, sample_size)`

**Behavior:**
- Loads the Multi-News test split from `Awesome075/multi_news_parquet`.
- Applies `Dataset.shuffle(seed=42).select(range(500))` for deterministic sampling.
- Initializes the `anchorsumpipeline` with the specified ablation flags.
- Iterates over the sampled dataset, saving JSON checkpoints every 50 samples to `data/ablations/<config_name>/`.
- Outputs: `data/ablations/<config_name>/ablation_<config_name>_final_500.json`

**Environment:** Requires `HF_TOKEN` to be set (via `.env` file or environment variable) for gated model access.

---

#### `ablations/Component_Ablation/run_all_sequential.py` — Component Ablation Runner

**Role:** Executes all four component ablation configurations sequentially on 500 samples each.

**Configurations:**

| Config Name | NLI | Entity | Revision | Corresponds to |
|:---|:---:|:---:|:---:|:---|
| `base` | ✗ | ✗ | ✗ | Unaugmented LLM generation |
| `no_nli` | ✗ | ✓ | ✓ | Entity Guard only |
| `no_entity` | ✓ | ✗ | ✓ | NLI audit only |
| `full` | ✓ | ✓ | ✓ | Complete AnchorSum (T<sub>max</sub>=1) |

**Usage:**
```bash
python ablations/Component_Ablation/run_all_sequential.py
```

**Output:** Generates summary JSON files in `data/ablations/{base, no_nli, no_entity, full}/`.

---

#### `ablations/Revision_Depth/revision2.py` — Revision Depth Runner

**Role:** Executes the full pipeline with T<sub>max</sub>=2 to produce the data for the verifier exploitation analysis (§5 of the paper).

**Usage:**
```bash
python ablations/Revision_Depth/revision2.py
```

**Output:** `data/ablations/full_revisions_2/ablation_full_revisions_2_final_500.json`

---

### Evaluation Scripts (`scripts/`)

Each evaluation script loads generated summaries from `data/ablations/`, computes the relevant metric against the original source documents, and writes per-instance detailed CSVs and summary statistics to `Results/`.

All evaluation scripts share a common structure:
1. Load the original Multi-News test dataset with identical seeded sampling.
2. Load the generated summaries from the corresponding ablation JSON file.
3. Match each generated summary to its source document via `example_id`.
4. Compute the metric and write results to CSV.

#### Component Ablation Evaluation Scripts (`scripts/Component_Ablation/`)

| Script | Metric | Backbone | Output Directory |
|:---|:---|:---|:---|
| `evaluate_summac_final.py` | SummaCConv | ViT-C NLI fine-tune | `Results/Component Ablation/summac_final_results/` |
| `evaluate_alignscore_simple.py` | AlignScore | RoBERTa-Large (7 NLU tasks) | `Results/Component Ablation/alignscore_results/` |
| `evaluate_bartscore_simple.py` | BARTScore (s→d, d→s) | BART-Large-CNN | `Results/Component Ablation/bartscore_simple_results/` |
| `evaluate_rouge_bertscore_simple.py` | ROUGE-1/2/L + BERTScore F₁ | n-gram / DeBERTa-XLarge-MNLI | `Results/Component Ablation/rouge_bert_simple_results/` |
| `evaluate_bertscore_xlarge.py` | BERTScore F₁ | DeBERTa-XLarge-MNLI | `Results/Component Ablation/bertscore_xlarge_results/` |
| `evaluate_unieval_fluency_simple.py` | UniEval Fluency | T5-Large evaluator | `Results/Component Ablation/unieval_fluency_results/` |

#### Revision Depth Evaluation Scripts (`scripts/Revision_Depth/`)

| Script | Metric | Output Directory |
|:---|:---|:---|
| `evaluate_summac_full_revisions_2.py` | SummaCConv | `Results/Revision Depth/summac_full_revisions_2_results/` |
| `evaluate_alignscore_full_revisions_2.py` | AlignScore | `Results/Revision Depth/alignscore_full_revisions_2_results/` |
| `evaluate_bartscore_full_revisions_2.py` | BARTScore (s→d, d→s) | `Results/Revision Depth/bartscore_full_revisions_2_results/` |
| `evaluate_rouge_bert_full_revisions_2.py` | ROUGE-1/2/L + BERTScore F₁ | `Results/Revision Depth/rouge_bert_full_revisions_2_results/` |
| `evaluate_unieval_fluency_full_revisions_2.py` | UniEval Fluency | `Results/Revision Depth/unieval_fluency_full_revisions_2_results/` |

**Usage (example):**
```bash
python scripts/Component_Ablation/evaluate_summac_final.py
python scripts/Revision_Depth/evaluate_bartscore_full_revisions_2.py
```

**Output format:** Each script produces:
- `<metric>_detailed_<config>.csv` — Per-instance scores (columns: `id`, `<metric>_score`, `method`)
- `<metric>_summary_<config>.csv` — Aggregate statistics (mean, median, std, min, max)
- `combined_summary.csv` — Cross-configuration summary (Component Ablation scripts only)

---

### Statistical Testing

#### `run_significance_testing.py` — Wilcoxon Signed-Rank Tests

**Role:** Performs paired, non-parametric significance tests on the precomputed per-instance metric scores. Produces the statistical evidence reported in Table 5 of the paper.

**Tests performed:**

| Test | Comparison | Metric | Alternative |
|:---|:---|:---|:---|
| Test 1 | Full (T<sub>max</sub>=1) vs Base | SummaCConv | Two-sided |
| Test 2 | Full (T<sub>max</sub>=1) vs Base | BARTScore s→d | Two-sided |
| Test 3 | T<sub>max</sub>=2 vs T<sub>max</sub>=1 | SummaCConv | Two-sided |
| Test 4 | T<sub>max</sub>=2 vs T<sub>max</sub>=1 | BARTScore s→d | Two-sided |
| Test 5 | no_entity vs Base | SummaCConv | Less |
| Test 6 | Full vs no_nli | BARTScore s→d | Greater |

**Usage:**
```bash
python run_significance_testing.py
```

**Output:** `Significance_Testing/wilcoxon_results.csv`

---

### Data Files (`data/`)

#### `data/multi_news_500_samples.json`

**Role:** Cached copy of the 500-sample Multi-News test subset used across all experiments. Generated via `Dataset.shuffle(seed=42).select(range(500))`.

**Format:** JSON array of objects with fields `document` (source articles, `|||`-delimited) and `summary` (reference summary).

**Purpose:** Ensures bitwise-identical sampling across experiment re-runs, even if the upstream Hugging Face dataset is updated.

#### `data/document.json`

**Role:** Sample output file containing generated summaries with multi-model comparisons. Useful for qualitative inspection and as input to the LLM-as-judge evaluation.

#### `data/ablations/`

**Role:** Contains the generated summaries for each ablation configuration.

| Subdirectory | Configuration | Pipeline Flags |
|:---|:---|:---|
| `base/` | Unaugmented LLM generation | `nli=False, entity=False, revision=False` |
| `no_nli/` | Entity Guard only | `nli=False, entity=True, revision=True` |
| `no_entity/` | NLI audit only | `nli=True, entity=False, revision=True` |
| `full/` | Complete AnchorSum (T<sub>max</sub>=1) | `nli=True, entity=True, revision=True` |
| `full_revisions_2/` | Complete AnchorSum (T<sub>max</sub>=2) | `nli=True, entity=True, revision=True, max_revisions=2` |

Each subdirectory contains checkpoint files (`checkpoint_<N>_samples.json`) and the final output (`ablation_<config>_final_500.json`).

**JSON record schema:**
```json
{
  "initial_draft": "...",
  "final_summary": "...",
  "reference": "...",
  "history": [
    {"revision": 0, "summary": "...", "flags": []},
    {"revision": 1, "summary": "...", "flags": ["..."]}
  ],
  "num_revisions": 1,
  "config_name": "full",
  "example_id": 0
}
```

---

### Precomputed Results (`Results/`)

All CSV files in the `Results/` directory contain the precomputed evaluation metric scores reported in the paper. These are provided so that results can be verified and statistical tests can be re-run without re-executing the full evaluation pipeline (which requires the external metric repositories and GPU resources).

Each metric subdirectory contains:
- **Per-instance detailed CSVs** (`<metric>_detailed_<config>.csv`): One row per evaluation instance with columns `id` and the metric-specific score column.
- **Summary statistics CSVs** (`<metric>_summary_<config>.csv`): Aggregate statistics (mean, median, std, min, max).
- **Combined summary** (`combined_summary.csv`): Cross-configuration comparison (Component Ablation only).

---

### Significance Testing Results (`Significance_Testing/`)

#### `Significance_Testing/wilcoxon_results.csv`

Precomputed output of `run_significance_testing.py`. Contains the Wilcoxon test statistic W, p-value, and mean values for each paired comparison.

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU with **≥ 16 GB VRAM** (≥ 8 GB with 4-bit NF4 fallback)
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

> **Note:** The external evaluation dependencies (AlignScore, BARTScore, SummaC, UniEval) are only required for re-running the evaluation scripts. The core AnchorSum pipeline (`src/`) operates independently.

---

## Reproducing Results

The full reproduction pipeline consists of three sequential stages. Precomputed outputs for all stages are provided in the repository, so each stage can be skipped if the upstream data is available.

### Step 1 — Generate Ablation Summaries

```bash
# Component ablation: base, no_nli, no_entity, full (T_max=1)
# Checkpoints every 50 samples; safe to interrupt and resume
python ablations/Component_Ablation/run_all_sequential.py

# Revision depth: full pipeline with T_max=2
python ablations/Revision_Depth/revision2.py
```

**Output:** `data/ablations/{base, no_nli, no_entity, full, full_revisions_2}/`

### Step 2 — Compute Evaluation Metrics

```bash
# Component Ablation metrics
python scripts/Component_Ablation/evaluate_summac_final.py
python scripts/Component_Ablation/evaluate_alignscore_simple.py
python scripts/Component_Ablation/evaluate_bartscore_simple.py
python scripts/Component_Ablation/evaluate_rouge_bertscore_simple.py
python scripts/Component_Ablation/evaluate_bertscore_xlarge.py
python scripts/Component_Ablation/evaluate_unieval_fluency_simple.py

# Revision Depth metrics
python scripts/Revision_Depth/evaluate_summac_full_revisions_2.py
python scripts/Revision_Depth/evaluate_bartscore_full_revisions_2.py
python scripts/Revision_Depth/evaluate_alignscore_full_revisions_2.py
python scripts/Revision_Depth/evaluate_rouge_bert_full_revisions_2.py
python scripts/Revision_Depth/evaluate_unieval_fluency_full_revisions_2.py
```

**Output:** `Results/{Component Ablation, Revision Depth}/`

### Step 3 — Run Statistical Significance Tests

```bash
python run_significance_testing.py
```

**Output:** `Significance_Testing/wilcoxon_results.csv`

### Reproducibility Parameters

| Parameter | Value |
|:---|:---|
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

All results below are reported over n = 498 Multi-News test instances (2 instances excluded due to GPU memory constraints). Full analysis, statistical tests, and discussion are presented in the paper.

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
| Full (T<sub>max</sub>=1) | 0.733 | **0.822** | **−4.019** | **0.384** | **0.611** | **0.944** |
| Full (T<sub>max</sub>=2) | **0.918** | 0.821 | −6.585 | 0.384 | 0.610 | 0.943 |
| **Δ (T₂ − T₁)** | **+0.185** | −0.001 | **−2.566** | <0.001 | −0.001 | −0.001 |

> The +0.185 SummaCConv gain at T<sub>max</sub>=2 is **verifier exploitation**, not genuine improvement. BARTScore s→d collapses by −2.566 nats on every single instance (W = 0, p = 2.68 × 10⁻⁸³), while AlignScore shows Δ < 0.001. See paper §5 for the full mechanistic analysis and three-condition detection protocol.

### LLM-as-Judge Evaluation (n = 50)

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

### Statistical Significance (Wilcoxon Signed-Rank Tests, n = 498)

| Test | W | p-value | Survives Bonferroni |
|:---|:---:|:---:|:---:|
| Full vs Base — SummaCConv | 26,687 | 4.49 × 10⁻²⁸ | ✓ |
| Full vs Base — BARTScore s→d | 50,753 | 4.01 × 10⁻⁴ | ✓ |
| T₂ vs T₁ — SummaCConv | 894 | 5.68 × 10⁻⁸¹ | ✓ |
| T₂ vs T₁ — BARTScore s→d | 0 | 2.68 × 10⁻⁸³ | ✓ |
| no_entity vs Base — SummaCConv | 43,895 | 2.91 × 10⁻⁵ | ✓ |

---

## Evaluation Metrics

| Metric | Backbone | Role in Analysis | What It Measures |
|:---|:---|:---|:---|
| **SummaCConv** | ViT-C NLI fine-tune | Primary faithfulness signal; also the auditing metric whose exploitation is studied | Sentence-level entailment consistency |
| **AlignScore** | RoBERTa-Large (7 NLU tasks) | Independent non-NLI faithfulness reference for cross-framework triangulation | Cross-framework factual alignment |
| **BARTScore s→d** | BART-large-CNN | Generative log-probability signal for register degradation detection | log P(summary \| source) — register-sensitive |
| **BARTScore d→s** | BART-large-CNN | Coverage signal (reverse direction) | log P(source \| summary) — coverage proxy |
| **ROUGE-1/2/L** | n-gram overlap | Reference-dependent lexical overlap; secondary metric | Token-level n-gram overlap with reference |
| **BERTScore F₁** | DeBERTa-XLarge-MNLI | Semantic similarity with reference; secondary metric | Token-level semantic similarity |
| **UniEval Fluency** | T5-Large evaluator | Surface quality control; confirms exploitation is invisible to fluency-based metrics | Linguistic fluency ∈ [0, 1] |

---

## Dataset

| Property | Value |
|:---|:---|
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
|:---|:---|:---|
| Base generation only | ~37 s | ~5.1 GPU-hours |
| Full pipeline (T<sub>max</sub>=1) | ~102 s | ~14.2 GPU-hours |
| Full pipeline (T<sub>max</sub>=2) | ~186 s | ~25.8 GPU-hours |

<sub>Benchmarked on a single NVIDIA A40 (45 GB HBM2).</sub>

---

## AI Use Disclosure

This project utilized AI assistants (including Anthropic Claude, Google Gemini, Kimi, and other AI tools) for code assistance, debugging, and help with evaluation of metrics. All core research contributions — algorithmic design, experimental methodology, results, and conclusions — are the original work of the author.
