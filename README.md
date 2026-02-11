# Extrisic calibration Lidar-camera

## Prerequisites

- **Python 3.10**
- **uv** (recommended) — [Astral’s uv](https://docs.astral.sh/uv/) for fast, reproducible installs

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

Important: The code assumed OpenCV BGR and did color[::-1] to get RGB. In my case the decoded images are already RGB, maybe the image have to be flip, otherwise you will se you calibration points blue.

## Clean up

To remove the environment and uv cache and start over:

```bash
deactivate
rm -rf .venv
uv cache clean
```
