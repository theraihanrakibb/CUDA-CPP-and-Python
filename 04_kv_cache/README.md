# 04 - KV Cache (C++ and Python)

**Paged KV cache management** for LLM-style attention, in both languages.

- `cpp/` - `kv_cache.h` (the paged manager) + `paged_attention.cu` (GPU kernel
  that reads K/V through the block table), verified against a CPU reference.
- `python/` - `kv_cache.py` - the same design with CuPy (RawKernel), verified
  against a NumPy reference.

Both demonstrate the core idea behind vLLM's PagedAttention: store cached
keys/values in fixed blocks addressed by a per-sequence **block table**
(virtual-memory style) instead of one big pre-allocated slot. This removes
memory fragmentation and lets requests share prompt blocks.

Start with `cpp/README.md` or `python/README.md` for build/run steps.
