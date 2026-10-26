# CUDA C++ and Python

A hands-on repo comparing **CUDA in C++** and **CUDA in Python** (CuPy), ending
in the thing everyone asks about after learning CUDA: **KV cache management for
LLM serving**. Code is written to read like a human wrote it - clear names,
CPU/golden references, and self-checks (gradient checks, max-error vs CPU).

> Hardware note: these files were authored on a machine **without** a CUDA
> toolkit or GPU, so they are *build-ready* but not compiled here. Build and
> run them on a machine that has the CUDA toolkit (for C++) or CuPy + a CUDA
> GPU (for Python). Each sub-folder has its own README + Makefile.

## Layout

| Folder | Language | What it teaches |
|--------|----------|-----------------|
| `01_basics/` | C++ | vector add, tiled matmul, shared-memory reduction |
| `02_benchmarks/` | C++ | memory bandwidth, occupancy, kernel timing |
| `03_applications/` | C++ | image blur, histogram, Monte-Carlo pi |
| `04_kv_cache/` | **C++ & Python** | a paged KV cache + paged-attention kernel (the vLLM idea) |
| `05_train_infer/` | **C++ & Python** | train a tiny transformer, then serve it from a KV cache |

The two starred folders are the core of the "KV cache + interface/training"
request: `04` is the **manager + GPU kernel**, `05` is the **end-to-end
train-then-serve story** that uses it.

## Build & run (C++)

Each C++ folder has a Makefile:

    cd 04_kv_cache/cpp
    make            # nvcc must be on PATH
    ./paged_attention

If your GPU isn't Pascal (sm_60), edit `CUDAFLAGS` in the Makefile, e.g.
`-arch=sm_75` (Turing), `-arch=sm_86` (Ampere), `-arch=sm_89` (Ada).

## Build & run (Python)

    cd 04_kv_cache/python
    pip install cupy-cuda12x        # pick the wheel for your CUDA version
    python kv_cache.py

    cd ../../05_train_infer/python
    pip install numpy cupy-cuda12x
    python tiny_transformer.py

## The KV cache idea in one paragraph

Transformers generate tokens one at a time; each step needs the attention
keys/values of *all* earlier tokens. Recomputing them is wasteful, so we store
them in a **KV cache**. Naively pre-allocating a max-length slot per request
wastes GPU memory (fragmentation). **Paged** management (vLLM) borrows virtual
memory: split the cache into fixed blocks, give each request a *block table*
that maps logical tokens -> scattered physical blocks, and grow one block at a
time. `04` implements this manager + the GPU attention kernel that reads
through the block table; `05` trains a model and then serves it this way.

## Pushing to GitHub

This was generated in an environment without `git`/GitHub access, so the push
is left to you. From this folder:

    git init
    git add .
    git commit -m "CUDA C++ and Python: basics, KV cache, train+infer"
    git branch -M main
    git remote add origin https://github.com/<you>/<repo>.git
    git push -u origin main

Or, to publish into an empty repo you already created on GitHub, just add the
remote and push as above. (Use a personal access token or SSH, not a password.)

## License

MIT - do what you like, just keep the attribution.
