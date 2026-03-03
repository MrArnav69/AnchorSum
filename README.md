<p align="center">
  <h1 align="center">AnchorSum</h1>
  <p align="center">
    <strong>Anchor-Guided Summarization with Iterative Verification for Faithful Multi-Document Synthesis</strong>
  </p>
  <p align="center">
    <a href="#overview">Overview</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#repository-structure">Repository Structure</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#ablation-studies">Ablation Studies</a> •
    <a href="#evaluation">Evaluation</a> •
    <a href="#results">Results</a> •
    <a href="#citation">Citation</a>
  </p>
</p>

---

## Overview

**AnchorSum** is a novel multi-document summarization (MDS) framework that addresses the critical challenge of factual hallucination in LLM-generated summaries. Rather than relying on a single forward pass, AnchorSum introduces a **generate → verify → revise** pipeline that anchors the summarization process to salient entities extracted from source documents and iteratively corrects factual inconsistencies through dual verification modules.

The framework leverages:

- **LLaMA 3.1-8B-Instruct** as the backbone generative model for single-pass multi-document summarization
- **Entity anchoring** via spaCy NER (`en_core_web_trf`) to identify and enforce coverage of key entities (persons, organizations, locations, dates, monetary values)
- **NLI-based factual verification** via DeBERTa-v3-Large (`cross-encoder/nli-deberta-v3-large`) to detect contradictory or unsupported claims at the sentence level
- **Iterative self-revision** where flagged errors are fed back to the LLM for targeted correction

AnchorSum is evaluated on **500 samples** from the [Multi-News](https://huggingface.co/datasets/Awesome075/multi_news_parquet) test split using a comprehensive suite of **six automatic evaluation metrics**, spanning content overlap, semantic similarity, factual consistency, and linguistic fluency.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AnchorSum Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────────────────────────┐      │
│  │ Source Docs   │────▶│  Entity Guard (spaCy NER)        │      │
│  │ (Multi-News)  │     │  Extract top-N salient anchors   │      │
│  └──────┬───────┘     └──────────────┬───────────────────┘      │
│         │                            │                          │
│         ▼                            ▼                          │
│  ┌──────────────────────────────────────────────────────┐       │
│  │          LLM Summarizer (LLaMA 3.1-8B-Instruct)     │       │
│  │   Generate initial draft with anchor constraints     │       │
│  └──────────────────────┬───────────────────────────────┘       │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Verification Loop (k iterations)         │       │
│  │                                                      │       │
│  │  ┌──────────────────┐   ┌─────────────────────────┐  │       │
│  │  │  NLI Verifier    │   │   Entity Guard          │  │       │
│  │  │  (DeBERTa-v3)    │   │   Coverage + Halluc.    │  │       │
│  │  │                  │   │   Detection             │  │       │
│  │  │  Sentence-level  │   │                         │  │       │
│  │  │  entailment      │   │  • Missing anchors      │  │       │
│  │  │  checking        │   │  • Hallucinated entities │  │       │
│  │  └────────┬─────────┘   └────────────┬────────────┘  │       │
│  │           │                          │               │       │
│  │           └──────────┬───────────────┘               │       │
│  │                      ▼                               │       │
│  │           ┌─────────────────────┐                    │       │
│  │           │   Flags Aggregated  │                    │       │
│  │           │   No flags → STOP   │                    │       │
│  │           │   Has flags → REVISE│                    │       │
│  │           └─────────┬───────────┘                    │       │
│  │                     │                                │       │
│  │                     ▼                                │       │
│  │           ┌──────────────────────────────────┐       │       │
│  │           │  LLM Revision Pass               │       │       │
│  │           │  Fix flagged errors, preserve     │       │       │
│  │           │  narrative coherence              │       │       │
│  │           └──────────────────────────────────┘       │       │
│  └──────────────────────────────────────────────────────┘       │
│                         │                                       │
│                         ▼                                       │
│              ┌───────────────────┐                               │
│              │  Final Summary    │                               │
│              └───────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
AnchorSum/
├── src/                              # Core framework source code
│   ├── __init__.py
│   ├── pipeline.py                   # Main AnchorSum pipeline orchestrator
│   ├── llm_summarizer.py            # LLaMA-based single-pass MDS with revision
│   └── verification/                 # Dual verification subsystem
│       ├── __init__.py
│       ├── nli_verifier.py           # NLI-based factual consistency checker (DeBERTa-v3)
│       └── entity_guard.py           # Entity anchor extraction & hallucination detector (spaCy)
│
├── ablations/                        # Ablation study experiment runners
│   ├── ablation_base_runner.py       # Shared experiment runner (dataset loading, checkpointing)
│   ├── Component_Ablation/
│   │   └── run_all_sequential.py     # Runs all 4 component ablation configs sequentially
│   └── Revision_Depth/
│       └── revision2.py              # Runs full pipeline with max_revisions=2
│
├── scripts/                          # Evaluation metric computation scripts
│   ├── Component_Ablation/
│   │   ├── evaluate_rouge_bertscore_simple.py    # ROUGE-1/2/L scoring
│   │   ├── evaluate_bertscore_xlarge.py          # BERTScore (DeBERTa-XLarge-MNLI)
│   │   ├── evaluate_alignscore_simple.py         # AlignScore (RoBERTa-Large, NLI-SP)
│   │   ├── evaluate_bartscore_simple.py          # BARTScore (BART-Large-CNN; bidirectional)
│   │   ├── evaluate_summac_final.py              # SummaC (SummaCConv with ViT-C)
│   │   └── evaluate_unieval_fluency_simple.py    # UniEval (fluency dimension)
│   └── Revision_Depth/
│       ├── evaluate_rouge_bert_full_revisions_2.py
│       ├── evaluate_bartscore_full_revisions_2.py
│       ├── evaluate_alignscore_full_revisions_2.py
│       ├── evaluate_summac_full_revisions_2.py
│       └── evaluate_unieval_fluency_full_revisions_2.py
│
├── data/                             # Datasets and generated ablation outputs
│   ├── multi_news_500_samples.json   # Cached 500-sample Multi-News test subset (seed=42)
│   ├── document.json                 # Sample outputs with multi-model comparisons
│   └── ablations/                    # Generated summaries per ablation configuration
│       ├── base/                     # No verification, no revision
│       ├── no_nli/                   # Entity Guard only (no NLI verification)
│       ├── no_entity/                # NLI Verifier only (no entity anchoring)
│       ├── full/                     # Full pipeline (1 revision)
│       └── full_revisions_2/         # Full pipeline (2 revisions)
│
├── Results/                          # Computed evaluation metrics
│   ├── Component Ablation/           # Metrics across 4 component ablation configs
│   │   ├── rouge_bert_simple_results/
│   │   ├── bertscore_xlarge_results/
│   │   ├── alignscore_results/
│   │   ├── bartscore_simple_results/
│   │   ├── summac_final_results/
│   │   └── unieval_fluency_results/
│   └── Revision Depth/              # Metrics for revision depth analysis (k=2)
│       ├── rouge_bert_full_revisions_2_results/
│       ├── summac_full_revisions_2_results/
│       └── unieval_fluency_full_revisions_2_results/
│
├── notebooks/                        # Jupyter notebooks (reserved)
├── requirements.txt                  # Python dependencies
├── anchorsum-6.pdf                   # Research paper
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended; 16GB+ VRAM for LLaMA 3.1-8B)
- [Hugging Face](https://huggingface.co/) account with access to `meta-llama/Llama-3.1-8B-Instruct`

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/AnchorSum.git
cd AnchorSum

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download the spaCy transformer model
python -m spacy download en_core_web_trf

# Set up your Hugging Face token
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

### Dependencies

| Package                 | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| `torch`                 | Deep learning backbone                     |
| `transformers`          | LLaMA 3.1 & DeBERTa model loading          |
| `accelerate`            | Efficient model distribution               |
| `bitsandbytes`          | Optional 4-bit quantization                |
| `datasets`              | Multi-News dataset loading                 |
| `spacy`                 | Named entity recognition (Entity Guard)    |
| `nltk`                  | Sentence tokenization for NLI verification |
| `rouge-score`           | ROUGE metric computation                   |
| `bert-score`            | BERTScore metric computation               |
| `sentence-transformers` | Semantic similarity utilities              |
| `scikit-learn`          | Statistical analysis utilities             |
| `python-dotenv`         | Environment variable management            |
| `pandas` / `numpy`      | Data processing and analysis               |
| `tqdm`                  | Progress tracking                          |

---

## Usage

### Basic Pipeline Inference

```python
from src.pipeline import anchorsumpipeline

# Initialize the full AnchorSum pipeline
pipeline = anchorsumpipeline(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    nli_model_name="cross-encoder/nli-deberta-v3-large",
    entity_model_name="en_core_web_trf",
    token="your_hf_token",
    max_revisions=1,       # Number of revision iterations
    nli=True,              # Enable NLI verification
    entity=True,           # Enable entity anchoring
    revision=True          # Enable iterative revision
)

# Process a multi-document input
documents = """Article 1: ... Article 2: ... Article 3: ..."""
result = pipeline.process(documents, reference_summary="optional reference")

print(result["final_summary"])
print(f"Revisions performed: {result['num_revisions']}")
print(f"Revision history: {result['history']}")
```

### Running Ablation Experiments

**Component Ablation** (4 configurations × 500 samples):

```bash
python ablations/Component_Ablation/run_all_sequential.py
```

This sequentially runs:

| Configuration | NLI | Entity | Revision | Description                               |
| :-----------: | :-: | :----: | :------: | ----------------------------------------- |
|    `base`     |  ✗  |   ✗    |    ✗     | Raw LLM generation (no verification)      |
|   `no_nli`    |  ✗  |   ✓    |    ✓     | Entity anchoring + revision, no NLI       |
|  `no_entity`  |  ✓  |   ✗    |    ✓     | NLI verification + revision, no anchoring |
|    `full`     |  ✓  |   ✓    |    ✓     | Complete AnchorSum pipeline               |

**Revision Depth Analysis** (max_revisions=2):

```bash
python ablations/Revision_Depth/revision2.py
```

### Running Evaluation Scripts

After generating ablation outputs, compute evaluation metrics:

```bash
# Component Ablation evaluations
python scripts/Component_Ablation/evaluate_rouge_bertscore_simple.py   # ROUGE-1/2/L
python scripts/Component_Ablation/evaluate_bertscore_xlarge.py         # BERTScore
python scripts/Component_Ablation/evaluate_alignscore_simple.py        # AlignScore
python scripts/Component_Ablation/evaluate_bartscore_simple.py         # BARTScore
python scripts/Component_Ablation/evaluate_summac_final.py             # SummaC
python scripts/Component_Ablation/evaluate_unieval_fluency_simple.py   # UniEval Fluency

# Revision Depth evaluations
python scripts/Revision_Depth/evaluate_rouge_bert_full_revisions_2.py
python scripts/Revision_Depth/evaluate_summac_full_revisions_2.py
python scripts/Revision_Depth/evaluate_unieval_fluency_full_revisions_2.py
```

---

## Ablation Studies

### Study I: Component Ablation

Systematically disables individual components to isolate their contributions:

| Configuration | Components Active             | Research Question                                            |
| ------------- | ----------------------------- | ------------------------------------------------------------ |
| **Base**      | LLM only                      | What is the raw generation quality without any verification? |
| **No NLI**    | LLM + Entity Guard + Revision | How much does NLI verification contribute?                   |
| **No Entity** | LLM + NLI Verifier + Revision | How critical is entity anchoring?                            |
| **Full**      | LLM + NLI + Entity + Revision | What is the combined effect of all components?               |

### Study II: Revision Depth

Evaluates the impact of increasing the maximum number of revision iterations from **k=1** to **k=2** using the full pipeline, investigating:

- Whether additional revision passes yield diminishing returns
- The trade-off between factual accuracy gains and computational cost
- The stability of the verification-revision feedback loop

---

## Evaluation Metric Setup

To run the evaluation scripts, you need to clone several external repositories into the respective script directories and download the required checkpoints.

### 1. Repository Dependencies

Clone the following repositories into **both** `scripts/Component_Ablation/` and `scripts/Revision_Depth/`:

- **AlignScore**: `https://github.com/yuh-zha/AlignScore`
- **BARTScore**: `https://github.com/neulab/BARTScore`
- **SummaC**: `https://github.com/tingofurro/summac` (Rename the cloned directory to `summac` if necessary)
- **UniEval**: `https://github.com/lukas-blecher/UniEval`

```bash
# Example for Component_Ablation
cd scripts/Component_Ablation
git clone https://github.com/yuh-zha/AlignScore
git clone https://github.com/neulab/BARTScore
git clone https://github.com/tingofurro/summac
git clone https://github.com/lukas-blecher/UniEval
```

### 2. AlignScore Checkpoint

The AlignScore evaluation requires the `AlignScore-large.ckpt` checkpoint. Download it and place it inside the `AlignScore/` directory within the script folders.

- **Download Link**: [AlignScore-large.ckpt](https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt)

```bash
# Example placement
# scripts/Component_Ablation/AlignScore/AlignScore-large.ckpt
# scripts/Revision_Depth/AlignScore/AlignScore-large.ckpt
```

---

## Evaluation

AnchorSum employs a comprehensive evaluation protocol spanning six complementary automatic metrics:

| Metric          | Model / Method                            | Dimension Measured         |
| --------------- | ----------------------------------------- | -------------------------- |
| **ROUGE-1/2/L** | Stemmed n-gram overlap                    | Content coverage (lexical) |
| **BERTScore**   | `microsoft/deberta-xlarge-mnli`           | Semantic similarity        |
| **AlignScore**  | `roberta-large` (NLI-SP mode)             | Factual alignment          |
| **BARTScore**   | `facebook/bart-large-cnn` (bidirectional) | Generation likelihood      |
| **SummaC**      | `SummaCConv` with ViT-C backbone          | Factual consistency        |
| **UniEval**     | T5-based multi-dimensional evaluator      | Linguistic fluency         |

All evaluations are conducted on **500 samples** from the Multi-News test split, randomly sampled with `seed=42` for reproducibility.

---

## Results

### Component Ablation Results (n=500)

#### Content Quality Metrics

| Configuration |  ROUGE-1  |  ROUGE-2  |  ROUGE-L  | BERTScore F1 |
| :------------ | :-------: | :-------: | :-------: | :----------: |
| Base          |   0.399   |   0.117   |   0.183   |    0.619     |
| No NLI        |   0.383   |   0.119   |   0.178   |    0.612     |
| No Entity     |   0.396   |   0.113   |   0.181   |    0.617     |
| **Full**      | **0.384** | **0.119** | **0.178** |  **0.611**   |

#### Factual Consistency Metrics

| Configuration | AlignScore | BARTScore (s→d) | BARTScore (d→s) |  SummaC   |
| :------------ | :--------: | :-------------: | :-------------: | :-------: |
| Base          |   0.828    |     -4.038      |     -3.470      |   0.715   |
| No NLI        |   0.824    |     -4.027      |     -3.481      |   0.733   |
| No Entity     |   0.829    |     -4.045      |     -3.487      |   0.712   |
| **Full**      | **0.822**  |   **-4.019**    |   **-3.469**    | **0.733** |

#### Fluency

| Configuration | UniEval Fluency |
| :------------ | :-------------: |
| Base          |      0.945      |
| No NLI        |      0.940      |
| No Entity     |      0.949      |
| **Full**      |    **0.944**    |

### Revision Depth Results (Full Pipeline, n=500)

| Revisions | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |  SummaC   | UniEval Fluency |
| :-------: | :-----: | :-----: | :-----: | :----------: | :-------: | :-------------: |
|    k=1    |  0.384  |  0.119  |  0.178  |    0.611     |   0.733   |      0.944      |
|    k=2    |  0.384  |  0.120  |  0.177  |    0.610     | **0.918** |      0.943      |

> **Key Finding:** Increasing revision depth from k=1 to k=2 yields a substantial improvement in factual consistency as measured by SummaC (**0.733 → 0.918**, +25.2%), while maintaining near-identical scores across content quality (ROUGE, BERTScore) and fluency metrics. This demonstrates that the verification-revision loop effectively targets and corrects factual errors without degrading overall summary quality.

---

## Core Modules

### `src/pipeline.py` — AnchorSum Pipeline Orchestrator

The central orchestrator that composes all components into the generate → verify → revise loop. Accepts configuration flags to enable/disable individual verification modules and controls revision depth.

### `src/llm_summarizer.py` — Single-Pass Multi-Document Summarizer

Wraps LLaMA 3.1-8B-Instruct with structured prompting for both initial draft generation (with optional entity anchor constraints) and targeted revision given audit flags. Uses greedy decoding (`do_sample=False`) with repetition penalty for deterministic, high-quality outputs.

### `src/verification/nli_verifier.py` — NLI Factual Verifier

Performs sentence-level Natural Language Inference between each summary sentence and the source documents using DeBERTa-v3-Large. Sentences classified as **CONTRADICTION** or **NEUTRAL** are flagged for revision.

### `src/verification/entity_guard.py` — Entity Anchor Guard

Dual-function module powered by spaCy NER:

1. **Anchor Extraction** — Identifies the top-N most frequent named entities (PERSON, ORG, GPE, LOC, DATE, MONEY, PERCENT) from source documents as mandatory coverage constraints
2. **Hallucination Detection** — Cross-references entities in the generated summary against the source to flag fabricated entities not present in the original documents

---

## Dataset

All experiments use the **Multi-News** dataset:

- **Source:** [Awesome075/multi_news_parquet](https://huggingface.co/datasets/Awesome075/multi_news_parquet)
- **Split:** Test
- **Sample Size:** 500 (randomly sampled, `seed=42`)
- **Task:** Multi-document summarization from 2–10 news articles per cluster

A cached version of the 500-sample subset is stored in `data/multi_news_500_samples.json` for reproducibility and offline use.

---

## Reproducibility

All experiments are fully reproducible with the following fixed parameters:

| Parameter      | Value                                     |
| -------------- | ----------------------------------------- |
| Random Seed    | `42`                                      |
| Sample Size    | `500`                                     |
| LLM            | `meta-llama/Llama-3.1-8B-Instruct`        |
| NLI Model      | `cross-encoder/nli-deberta-v3-large`      |
| NER Model      | `en_core_web_trf`                         |
| Precision      | `float32` (full precision)                |
| Generation     | Greedy decoding, `repetition_penalty=1.1` |
| Max New Tokens | `1024`                                    |
| Checkpointing  | Every 50 samples                          |

---

## License

This project is provided for academic and research purposes. Please refer to the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with LLaMA 3.1 · DeBERTa-v3 · spaCy · Evaluated with ROUGE, BERTScore, AlignScore, BARTScore, SummaC & UniEval</sub>
</p>
