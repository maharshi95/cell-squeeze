set -e

DEFAULT_ENV_NAME=
env_name="${1:-genomics}"

if ! command -v conda &>/dev/null; then
    echo "'conda' not found; install it here: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if ! { conda env list | grep "${env_name}"; } >/dev/null 2>&1; then
    conda create -y -n "${env_name}" python=3.10
fi

eval "$(conda shell.bash hook)"
conda activate genomics

echo "Python Interpreter: $(which python)"

# Data science packages
pip install --no-cache numpy pandas scipy einops scikit-learn matplotlib jupyterlab

pip install --no-cache BitVector

pip install --no-cache rich loguru

# XLSX support for pandas
pip install --no-cache xlrd openpyxl
