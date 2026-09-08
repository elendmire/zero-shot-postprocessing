#!/usr/bin/env bash
# Run this ON THE SSH SERVER once host/credentials are known. Not runnable locally —
# installs the heavy TSFM/baseline stack this laptop scaffold intentionally excludes.
set -euo pipefail

mkdir -p ~/zeropp && cd ~/zeropp
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate

uv pip install "timesfm[torch]" numpy pandas xarray netcdf4 zarr \
  scikit-learn scipy properscoring pyarrow matplotlib hydra-core
uv pip install statsmodels torch
uv pip install climetlab climetlab-eumetnet-postprocessing-benchmark

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
