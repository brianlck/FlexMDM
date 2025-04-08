### Extra installation instruction
To use flash attention, what may have to do the following
```
uv pip install torch setuptools
uv sync
uv add flash-attn --no-build-isolation
```
due the FlashAttention using deprecated build tools