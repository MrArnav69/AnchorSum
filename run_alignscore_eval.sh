#!/bin/bash

# Script to run AlignScore evaluation in conda environment
echo "🔬 Starting AlignScore evaluation for full_revisions_2..."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alignscore_eval

# Check if environment is activated
if [ "$CONDA_DEFAULT_ENV" != "alignscore_eval" ]; then
    echo "❌ Failed to activate conda environment 'alignscore_eval'"
    echo "🔧 Please run setup_alignscore_conda.sh first"
    exit 1
fi

echo "✅ Conda environment 'alignscore_eval' activated"
echo "🏃 Running AlignScore evaluation..."

# Run the evaluation
python evaluate_alignscore_full_revisions_2.py

echo "🎉 Evaluation completed!"
