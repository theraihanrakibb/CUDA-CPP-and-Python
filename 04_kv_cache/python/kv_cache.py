"""kv_cache.py - paged KV cache in CUDA Python (CuPy).

Same idea as the C++ version: a manager owns one big GPU buffer, hands out
fixed-size blocks, and a GPU kernel reads K/V back through a per-sequence
block table to compute self-attention. Verified against a NumPy reference.

Requires: cupy  (pip install cupy-cuda12x   # match your CUDA version)
"""
import numpy as np
import cupy as cp

BLOCK_SIZE = 4


class PagedKVCache:
    """CPU-side bookkeeping + one big GPU buffer of paged K/V storage."""

    def __init__(self, num_blocks, head_dim, num_heads=1):
        self.block_size = BLOCK_SIZE
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_dim = head_dim
        # physical storage: [num_blocks][block_size][num_heads][head_dim]
        self.d_K = cp.zeros((num_blocks, BLOCK_SIZE, num_heads, head_dim), dtype=cp.float32)
        self.d_V = cp.zeros((num_blocks, BLOCK_SIZE, num_heads, head_dim), dtype=cp.float32)
        self.block_tables = []   # per sequence -> list of physical block ids
        self.seq_len = []
        self.free_blocks = list(range(num_blocks - 1, -1, -1))

    def new_sequence(self):
        sid = len(self.block_tables)
        self.block_tables.append([])
        self.seq_len.append(0)
        return sid

    def ensure_block(self, sid, pos):
        need = pos // self.block_size + 1
        while len(self.block_tables[sid]) < need:
            self.block_tables[sid].append(self.free_blocks.pop())

    def append(self, sid, pos, k, v):
        """k, v: host or device arrays of shape [num_heads, head_dim]."""
        self.ensure_block(sid, pos)
        pb = self.block_tables[sid][pos // self.block_size]
        slot = pos % self.block_size
        self.d_K[pb, slot] = cp.asarray(k, dtype=cp.float32)
        self.d_V[pb, slot] = cp.asarray(v, dtype=cp.float32)
        self.seq_len[sid] = max(self.seq_len[sid], pos + 1)

    def dense(self, sid):
        """Return dense K, V of shape [L, num_heads, head_dim] for checking."""
        L = self.seq_len[sid]
        K = np.zeros((L, self.num_heads, self.head_dim), dtype=np.float32)
        V = np.zeros((L, self.num_heads, self.head_dim), dtype=np.float32)
        for pos in range(L):
            pb = self.block_tables[sid][pos // self.block_size]
            slot = pos % self.block_size
            K[pos] = cp.asnumpy(self.d_K[pb, slot])
            V[pos] = cp.asnumpy(self.d_V[pb, slot])
        return K, V


# ---- GPU kernel: one block per query position, thread 0 does the work ----
KERNEL = r"""
extern "C" __global__ void paged_attention(
    const float* K, const float* V, const int* block_table,
    int L, int block_size, int head_dim, const float* Q, float* Out, int qpos) {
    if (threadIdx.x != 0) return;
    extern __shared__ float scores[];
    float maxs = -1e30f;
    for (int j = 0; j < L; ++j) {
        int b = block_table[j / block_size];
        int slot = j % block_size;
        const float* k = K + ((b * block_size + slot) * head_dim);
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d) s += Q[d] * k[d];
        scores[j] = s;
        if (s > maxs) maxs = s;
    }
    float sum = 0.0f;
    for (int j = 0; j < L; ++j) { scores[j] = expf(scores[j] - maxs); sum += scores[j]; }
    for (int d = 0; d < head_dim; ++d) Out[qpos * head_dim + d] = 0.0f;
    for (int j = 0; j < L; ++j) {
        int b = block_table[j / block_size];
        int slot = j % block_size;
        const float* v = V + ((b * block_size + slot) * head_dim);
        float w = scores[j] / sum;
        for (int d = 0; d < head_dim; ++d) Out[qpos * head_dim + d] += w * v[d];
    }
}
"""
_paged_attention = cp.RawKernel(KERNEL, "paged_attention")


def run_paged_attention(cache, sid):
    L = cache.seq_len[sid]
    head_dim = cache.head_dim
    bt = cp.asarray(cache.block_tables[sid], dtype=cp.int32)
    K_flat = cache.d_K.reshape(-1, head_dim)        # [num_blocks*block_size, head_dim]
    V_flat = cache.d_V.reshape(-1, head_dim)
    # build dense query = K (self-attention demo)
    Kd, _ = cache.dense(sid)
    out = cp.zeros((L, head_dim), dtype=cp.float32)
    for q in range(L):
        Q = cp.asarray(Kd[q, 0], dtype=cp.float32)   # head 0
        _paged_attention(
            (1,), (1,),
            (K_flat, V_flat, bt, L, cache.block_size, head_dim,
             Q, out, q),
            shared_mem=L * 4)
    return cp.asnumpy(out)


def reference_attention(K, V, head_dim):
    L = K.shape[0]
    out = np.zeros((L, head_dim), dtype=np.float32)
    for q in range(L):
        Q = K[q, 0]
        s = (K[:, 0] * Q).sum(axis=-1)
        s -= s.max()
        w = np.exp(s); w /= w.sum()
        out[q] = (w[:, None] * V[:, 0]).sum(axis=0)
    return out


def main():
    L, head_dim, num_heads = 10, 8, 1
    cache = PagedKVCache(num_blocks=32, head_dim=head_dim, num_heads=num_heads)
    sid = cache.new_sequence()
    rng = np.random.default_rng(42)
    Kh = rng.standard_normal((L, num_heads, head_dim)).astype(np.float32)
    Vh = rng.standard_normal((L, num_heads, head_dim)).astype(np.float32)
    for t in range(L):
        cache.append(sid, t, Kh[t], Vh[t])

    gpu_out = run_paged_attention(cache, sid)
    ref_out = reference_attention(Kh, Vh, head_dim)
    err = np.abs(gpu_out - ref_out).max()
    print(f"sequence length L={L}, blocks used={len(cache.block_tables[sid])}/32")
    print(f"max error vs NumPy reference: {err:.3e}")


if __name__ == "__main__":
    main()
