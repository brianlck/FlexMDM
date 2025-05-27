### Dependency management
This repository uses the uv package manager
To install uv pleaes refer to the official uv website and make sure to add the uv binary to $PATH.

Installing dependency is as simple as,
```
uv sync
```

The Flash Attention build script is not compatible with uv and require you CUDA 11.4+

To properly install Flash Attentoin:

1. Load CUDA 
```
module load cuda/12.4.1-fasrc01
```

2. Install FlashAttention build script dependencies and build with --no-build-isolation
```
uv pip install torch setuptools
uv add flash-attn --no-build-isolation
uv sync
```


### Instructions for using VSCode Jupyter Notebook for testing
Notebooks are provided to play with the sampling algorithms with trained models.
To use VSCode Jupyter with GPUs, the easiest way is to login to a compute node and establish a connection tunnel.
This is very easy to do with the VSCode server module on the FASRC cluster,

```
salloc [GPU Things]
module load vscode
code tunnel
```

### Pre-commit Hook
The codebase employ a ruff pre-commit hook for style fomartting.

After you've installed the necessary dependencies, install the pre-commit hooks by,

```
pre-commit install
```

