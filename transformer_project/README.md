# Transformer Project

A transformer implementation for German-English translation using the WMT17 dataset.

## Project Structure

```
transformer_project/
├── __init__.py              # Root package exports
├── pyproject.toml           # Poetry configuration
├── optimization.py          # LR Scheduler and optimizer
├── dataset.py               # Translation dataset class
├── configs/                 # YAML configuration files
│   ├── wmt17_sinusoidal.yaml
│   └── wmt17_rope.yaml
├── modelling/               # Model architecture
│   ├── __init__.py
│   ├── attention.py
│   ├── embedding.py
│   ├── feedforward.py
│   ├── functional.py
│   ├── layer_norm.py
│   ├── positional_encoding.py
│   └── transformer.py
├── run/                     # Training scripts
│   └── train_wmt.py
└── text/                    # Tokenizer module
    ├── __init__.py
    └── transformer_tokenizer.py
```

## Setup with Poetry

### What is Poetry?

Poetry is a modern Python dependency management and packaging tool. It handles:
- **Virtual environments**: Automatically creates isolated Python environments
- **Dependencies**: Manages your project's dependencies via `pyproject.toml`
- **Packaging**: Makes it easy to build and publish packages
- **Lock files**: `poetry.lock` ensures reproducible installs across machines

### Installing Poetry

```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Or with pipx (recommended)
pipx install poetry
```

### Setting Up This Project

1. **Navigate to the project directory:**
   ```bash
   cd /home/tosa098h/infera-abtin/test/transformer_project
   ```

2. **Set Python version (if needed):**
   Poetry requires Python 3.11+. If your default Python is older:
   ```bash
   poetry env use /path/to/python3.11
   ```

3. **Install dependencies:**
   ```bash
   poetry install
   ```
   This creates a virtual environment and installs all dependencies from `pyproject.toml`.

4. **Activate the environment (optional):**
   ```bash
   poetry shell
   ```
   Or run commands directly with `poetry run`.

### Common Poetry Commands

| Command | Description |
|---------|-------------|
| `poetry install` | Install dependencies from pyproject.toml |
| `poetry add <package>` | Add a new dependency |
| `poetry remove <package>` | Remove a dependency |
| `poetry update` | Update all dependencies |
| `poetry shell` | Activate the virtual environment |
| `poetry run <command>` | Run a command in the virtual environment |
| `poetry env info` | Show environment information |
| `poetry env list` | List all virtual environments |

## Training

### Run with default sinusoidal positional encoding:

```bash
cd /home/tosa098h/infera-abtin/test/transformer_project
poetry run python run/train_wmt.py --config configs/wmt17_sinusoidal.yaml
```

### Run with RoPE positional encoding:

```bash
poetry run python run/train_wmt.py --config configs/wmt17_rope.yaml
```

### Command-line options:

```bash
poetry run python run/train_wmt.py \
    --config configs/wmt17_sinusoidal.yaml \
    --epochs 10 \
    --resume_from ./weights/epoch_5.pt \
    --start_epoch 5
```

## Configuration

Edit the YAML config files in `configs/` to customize:
- `batch_size`, `epochs`, `learning_rate`
- `d_model`, `n_heads`, `num_encoder_layers`, etc.
- `positional_encoding_type`: "sinusoidal" or "rope"
- `use_wandb`: true/false for Weights & Biases logging
- `weights_dir`: where to save checkpoints

## Key Differences from pyproject.toml

The `pyproject.toml` file replaces:
- `requirements.txt` → dependencies listed in `[project.dependencies]`
- `setup.py` → build config in `[build-system]`
- Package metadata is all in one place

Example `pyproject.toml`:
```toml
[project]
name = "transformer_project"
version = "0.1.0"
requires-python = ">=3.11,<3.15"
dependencies = [
    "torch (>=2.10.0,<3.0.0)",
    "datasets (>=4.5.0,<5.0.0)",
    # ... more deps
]

[tool.poetry]
packages = [
    {include = "modelling"},
    {include = "text"}
]

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
```
