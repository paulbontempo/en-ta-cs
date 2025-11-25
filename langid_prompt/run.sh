#!/bin/bash
#SBATCH --partition=blanca-blast-lecs
#SBATCH --account=blanca-blast-lecs
#SBATCH --qos=blanca-blast-lecs
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100_2g.20gb:1
#SBATCH --time=00:20:00
#SBATCH --job-name=test-langid
#SBATCH --output=output-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pabo8622@colorado.edu

# Load modules
module purge
module load anaconda/2023.09

# Set up project directories BEFORE conda
export CONDA_ENVS_PATH=/projects/pabo8622/conda/envs
export CONDA_PKGS_DIRS=/projects/pabo8622/conda/pkgs
export PIP_PREFIX=/projects/pabo8622/python_packages
export PYTHONPATH=/projects/pabo8622/python_packages/lib/python3.11/site-packages:$PYTHONPATH
export PIP_NO_USER=1
export PYTHONNOUSERSITE=1

# Create necessary directories
mkdir -p $CONDA_ENVS_PATH
mkdir -p $CONDA_PKGS_DIRS
mkdir -p $PIP_PREFIX

# Set up HuggingFace cache directory
export HF_HOME=/projects/pabo8622/.cache/huggingface
export TRANSFORMERS_CACHE=/projects/pabo8622/.cache/huggingface/transformers
mkdir -p $HF_HOME

# Read token from home directory and set it explicitly
if [ -f ~/.cache/huggingface/token ]; then
    export HF_TOKEN=$(cat ~/.cache/huggingface/token | tr -d '[:space:]')
    echo "HuggingFace token loaded"
else
    echo "WARNING: No HuggingFace token found at ~/.cache/huggingface/token"
fi

# Create conda environment if it doesn't exist
if [ ! -d "$CONDA_ENVS_PATH/langid" ]; then
    echo "Creating conda environment in projects directory..."
    conda create -p $CONDA_ENVS_PATH/langid python=3.11 -y
fi

# Activate conda environment
conda activate $CONDA_ENVS_PATH/langid

# Install dependencies with pip to projects directory
echo "Installing dependencies..."
python -m pip install --no-user --prefix=$PIP_PREFIX --ignore-installed "numpy<2" torch transformers pandas tqdm accelerate huggingface-hub

# Print environment info
echo "Python version:"
python --version
echo "Python path:"
which python
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run the language ID classification script
echo "Starting language ID classification..."
python llm_langid.py

echo "Job completed!"