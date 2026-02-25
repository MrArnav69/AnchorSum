# AnchorSum Ablation Evaluation Scripts

This repository contains 5 comprehensive evaluation scripts for analyzing ablation study results. Each script calculates different metrics and outputs both detailed per-sample results and summary statistics in CSV format.

## Available Evaluation Scripts

### 1. BARTScore Evaluation (`evaluate_bartscore.py`)
- **Metrics**: BARTScore (sum2doc, doc2sum, and average)
- **Model**: facebook/bart-large-cnn
- **Output**: 
  - `bartscore_detailed_{config}.csv` - Per-sample scores
  - `bartscore_summary_{config}.csv` - Summary statistics
  - `bartscore_combined_summary.csv` - Combined summary for all configs

### 2. UniEval Fluency Evaluation (`evaluate_unieval_fluency.py`)
- **Metrics**: UniEval Fluency scores only
- **Model**: UniEval pretrained model
- **Output**:
  - `unieval_fluency_detailed_{config}.csv` - Per-sample fluency scores
  - `unieval_fluency_summary_{config}.csv` - Summary statistics
  - `unieval_fluency_combined_summary.csv` - Combined summary

### 3. AlignScore Evaluation (`evaluate_alignscore.py`)
- **Metrics**: AlignScore factual consistency scores
- **Model**: AlignScore-large (best performing model)
- **Checkpoint**: Uses the downloaded AlignScore-large.ckpt
- **Output**:
  - `alignscore_detailed_{config}.csv` - Per-sample alignment scores
  - `alignscore_summary_{config}.csv` - Summary statistics
  - `alignscore_combined_summary.csv` - Combined summary

### 4. SummaC Evaluation (`evaluate_summac.py`)
- **Metrics**: SummaC consistency scores
- **Model**: SummaC-conv-vitc (best performing model)
- **Output**:
  - `summac_detailed_{config}.csv` - Per-sample consistency scores
  - `summac_summary_{config}.csv` - Summary statistics
  - `summac_combined_summary.csv` - Combined summary

### 5. ROUGE and BERTScore Evaluation (`evaluate_rouge_bertscore.py`)
- **Metrics**: 
  - ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1)
  - BERTScore (precision, recall, F1)
- **Models**: 
  - ROUGE: Standard ROUGE implementation
  - BERTScore: microsoft/deberta-xlarge-mnli
- **Output**:
  - `rouge_bertscore_detailed_{config}.csv` - Per-sample scores
  - `rouge_bertscore_summary_{config}.csv` - Summary statistics
  - `rouge_bertscore_combined_summary.csv` - Combined summary

## Input Data Format

The scripts expect ablation data in JSON format with the following structure:
```json
[
  {
    "final_summary": "Generated summary text...",
    "reference": "Reference document text...",
    "config_name": "base",
    "example_id": 0,
    "anchors": [],
    "revisions": 0,
    "history": [...]
  }
]
```

## Supported Ablation Configurations

The scripts are configured to evaluate the following ablation files:
- `data/ablations/base/ablation_base_final_500.json`
- `data/ablations/no_nli/ablation_no_nli_final_500.json`
- `data/ablations/no_entity/ablation_no_entity_final_500.json`
- `data/ablations/full/ablation_full_final_500.json` (to be created)

## GPU Optimization

All scripts are optimized for A40 GPU with:
- **BARTScore**: Batch size 16
- **UniEval**: Batch size 16
- **AlignScore**: Batch size 32
- **SummaC**: Batch size 16
- **ROUGE/BERTScore**: Batch size 32

## Installation

1. Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. Install evaluation dependencies:
```bash
pip install -r evaluation_requirements.txt
```

3. Install spaCy model for UniEval:
```bash
python -m spacy download en_core_web_sm
```

4. Install additional dependencies for each library:
```bash
# BARTScore (already included in repo)
# UniEval (already included in repo)
# AlignScore (already included in repo)
# SummaC (already included in repo)
```

## Usage

### Running Individual Evaluations

Run each script separately:
```bash
python evaluate_bartscore.py
python evaluate_unieval_fluency.py
python evaluate_alignscore.py
python evaluate_summac.py
python evaluate_rouge_bertscore.py
```

### Running All Evaluations

Use the master script to run all evaluations:
```bash
python run_all_evaluations.py
```

## Output Structure

All results are saved to `/workspace/AnchorSum/evaluation_results/` with the following structure:
```
evaluation_results/
├── bartscore/
│   ├── bartscore_detailed_base.csv
│   ├── bartscore_summary_base.csv
│   ├── bartscore_detailed_no_nli.csv
│   ├── bartscore_summary_no_nli.csv
│   ├── bartscore_detailed_no_entity.csv
│   ├── bartscore_summary_no_entity.csv
│   └── bartscore_combined_summary.csv
├── unieval_fluency/
│   └── ...
├── alignscore/
│   └── ...
├── summac/
│   └── ...
└── rouge_bertscore/
    └── ...
```

## Detailed Output Format

Each detailed CSV contains:
- `example_id`: Sample identifier
- `config_name`: Ablation configuration name
- `summary`: Generated summary text
- `document`: Reference document text
- Various metric scores depending on the evaluation script

## Summary Statistics Format

Each summary CSV contains statistical measures for each metric:
- Mean
- Median
- Standard deviation
- Minimum
- Maximum
- 25th percentile (Q25)
- 75th percentile (Q75)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch sizes in the scripts
2. **Missing Dependencies**: Ensure all requirements are installed
3. **Model Download Issues**: Check internet connection and HuggingFace access
4. **File Not Found**: Verify ablation file paths exist

### Memory Optimization

If you encounter memory issues on A40 GPU:
1. Reduce batch sizes in the script configurations
2. Close other GPU processes
3. Use gradient checkpointing if available

## Performance Notes

- AlignScore-large checkpoint is ~4.6GB and requires significant GPU memory
- BERTScore with deberta-xlarge-mnli is computationally intensive
- SummaC-conv-vitc requires substantial memory for large documents
- All scripts include error handling for batch processing failures

## Citation

When using these evaluation scripts, please cite the respective papers for each metric:
- BARTScore: [BARTScore: Evaluating Generated Text by Text Generation](https://arxiv.org/abs/2104.08627)
- UniEval: [UniEval: A Unified Evaluation Framework for Natural Language Generation](https://arxiv.org/abs/2212.10505)
- AlignScore: [AlignScore: Evaluating Factual Consistency with a Unified Alignment Function](https://arxiv.org/abs/2305.16739)
- SummaC: [SummaC: Re-Visiting NLI-based Fact-Checking for Summarization Evaluation](https://arxiv.org/abs/2111.09525)
- ROUGE: [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)
- BERTScore: [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
