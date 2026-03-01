#!/bin/bash

# Conda environment setup for AlignScore evaluation
echo "🔧 Setting up conda environment for AlignScore evaluation..."

# Create new conda environment
conda create -n alignscore_eval python=3.9 -y

# Activate environment
echo "📦 Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alignscore_eval

# Install PyTorch with CUDA support
echo "🚀 Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install required packages
echo "📚 Installing required packages..."
pip install pandas numpy tqdm datasets transformers

# Install AlignScore dependencies
echo "🔬 Installing AlignScore dependencies..."
cd AlignScore
pip install -e .
cd ..

echo "✅ Conda environment 'alignscore_eval' is ready!"
echo "🎯 To activate: conda activate alignscore_eval"
echo "🏃 To run evaluation: python evaluate_alignscore_full_revisions_2.py"
