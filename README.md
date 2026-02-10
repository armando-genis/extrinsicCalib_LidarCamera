# multiview-lane-fusion

Multi-view Lane Fusion Library — lane detection and fusion across multiple camera views.

## Prerequisites

- **Python 3.10**
- **uv** (recommended) — [Astral’s uv](https://docs.astral.sh/uv/) for fast, reproducible installs
- **CUDA 12.1** (optional, for GPU) — install steps below use CUDA 12.1 builds

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add uv to your PATH and reload your shell:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
uv --version
```

### 2. Create and activate a virtual environment

```bash
uv venv .venv_extrinsic --python 3.10

source .venv_extrinsic/bin/activate
```

On Windows (PowerShell): `.venv\Scripts\activate`

### 3. Install dependencies (CUDA 12.1 / GPU)

For GPU support with CUDA 12.1, install in this order:

```bash

```

**CPU-only or after syncing from lockfile:** from the project root run:

```bash
uv sync
```

### 4. Verify the installation

Basic stack check:

```bash
python - << 'EOF'
import torch, mmcv
from mmcv.ops import batched_nms
print("Torch:", torch.__version__)
print("MMCV:", mmcv.__version__)
print("batched_nms OK")
EOF
```

Full stack (including MMEngine and MMDet):

```bash
python - << 'EOF'
import torch, mmcv, mmengine, mmdet
print("Torch:", torch.__version__)
print("MMCV:", mmcv.__version__)
print("MMEngine:", mmengine.__version__)
print("MMDet:", mmdet.__version__)
EOF
```

Check that DCNv2 (Deformable Convolution) is available:

```bash
python - << 'EOF'
from mmcv.ops import DeformConv2d
print("DCNv2 available")
EOF
```

## Run

```bash
uv run --no-sync python backbone_builder.py
```

With venv already activated:

```bash
python backbone_builder.py
```

## Clean up

To remove the environment and uv cache and start over:

```bash
deactivate
rm -rf .venv
uv cache clean
```
