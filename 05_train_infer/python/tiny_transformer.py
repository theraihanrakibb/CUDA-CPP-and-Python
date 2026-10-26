"""tiny_transformer.py - train a tiny transformer (NumPy), then serve
it through a paged KV cache on the GPU (CuPy).

CUDA-Python twin of the C++ project in ./cpp. The interesting part - reading
cached K/V back through a block table on the GPU - uses a CuPy RawKernel,
exactly like project 04.

Requirements: numpy, cupy-cuda12x
"""
import numpy as np
import cupy as cp

D, VOCAB, T = 8, 12, 8
rng = np.random.default_rng(1)


class TinyTransformer:
    def __init__(self):
        s = 0.1
        self.E = rng.standard_normal((VOCAB, D)).astype(np.float32) * s
        self.Wq = rng.standard_normal((D, D)).astype(np.float32) * s
        self.Wk = rng.standard_normal((D, D)).astype(np.float32) * s
        self.Wv = rng.standard_normal((D, D)).astype(np.float32) * s
        self.Wo = rng.standard_normal((D, D)).astype(np.float32) * s
        self.Wc = rng.standard_normal((D, VOCAB)).astype(np.float32) * s
        self.gE = np.zeros_like(self.E)
        self.gWq = np.zeros_like(self.Wq)
        self.gWk = np.zeros_like(self.Wk)
        self.gWv = np.zeros_like(self.Wv)
        self.gWo = np.zeros_like(self.Wo)
        self.gWc = np.zeros_like(self.Wc)

    def forward(self, seq):
        X = self.E[seq]                       # (T,D)
        Q = X @ self.Wq.T
        K = X @ self.Wk.T
        V = X @ self.Wv.T
        scores = (Q @ K.T) / np.sqrt(D).astype(np.float32)
        Tt = scores.shape[0]
        mask = np.triu(np.ones((Tt, Tt), dtype=bool), 1)
        scores[mask] = -1e30
        scores -= scores.max(-1, keepdims=True)
        a = np.exp(scores); a /= a.sum(-1, keepdims=True)
        C = a @ V
        H = C @ self.Wo.T
        LOGITS = H @ self.Wc          # (T,D) @ (D,VOCAB) -> (T,VOCAB)
        return X, Q, K, V, a, C, H, LOGITS

    def loss(self, LOGITS, seq):
        P = len(seq) - 1
        target = np.array(seq[1:])
        logp = LOGITS[:P] - LOGITS[:P].max(-1, keepdims=True)
        logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
        return float(-logp[np.arange(P), target].mean())

    def backward(self, seq, X, Q, K, V, a, C, H, LOGITS):
        P = len(seq) - 1
        glogits = np.exp(LOGITS - LOGITS.max(-1, keepdims=True))
        glogits /= glogits.sum(-1, keepdims=True)
        target = np.array(seq[1:])
        glogits[:P, target] -= 1.0
        glogits /= P

        self.gWc = H.T @ glogits
        gh = glogits @ self.Wc.T   # (T,VOCAB) @ (VOCAB,D) -> (T,D)
        self.gWo = C.T @ gh
        gc = gh @ self.Wo
        gV = a.T @ gc
        gv_inner = gc @ V.T
        gscore = a * (gv_inner - (a * gv_inner).sum(-1, keepdims=True))
        gQ = gscore @ K / np.sqrt(D)
        gK = gscore.T @ Q / np.sqrt(D)
        self.gWq = X.T @ gQ
        self.gWk = X.T @ gK
        self.gWv = X.T @ gV
        gx = gQ + gK + gV
        self.gE = np.zeros_like(self.E)
        for t, tok in enumerate(seq):
            self.gE[tok] += gx[t]

    def sgd(self, lr):
        self.E -= lr * self.gE
        self.Wq -= lr * self.gWq
        self.Wk -= lr * self.gWk
        self.Wv -= lr * self.gWv
        self.Wo -= lr * self.gWo
        self.Wc -= lr * self.gWc

    def token_kv(self, tok):
        x = self.E[tok]
        return x @ self.Wk.T, x @ self.Wv.T


# ---- paged KV cache (CuPy) + paged-attention context kernel ----
BLOCK_SIZE = 4
KERNEL = r"""
extern "C" __global__ void pagedAttentionContext(
    const float* K, const float* V, const int* block_table,
    int L, int block_size, int head_dim, const float* Q, float* Ctx, int qpos) {
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
    for (int d = 0; d < head_dim; ++d) Ctx[qpos * head_dim + d] = 0.0f;
    for (int j = 0; j < L; ++j) {
        int b = block_table[j / block_size];
        int slot = j % block_size;
        const float* v = V + ((b * block_size + slot) * head_dim);
        float w = scores[j] / sum;
        for (int d = 0; d < head_dim; ++d) Ctx[qpos * head_dim + d] += w * v[d];
    }
}
"""
_paged_ctx = cp.RawKernel(KERNEL, "pagedAttentionContext")


class PagedKVCache:
    def __init__(self, num_blocks, head_dim):
        self.block_size = BLOCK_SIZE
        self.head_dim = head_dim
        self.d_K = cp.zeros((num_blocks, BLOCK_SIZE, head_dim), dtype=cp.float32)
        self.d_V = cp.zeros((num_blocks, BLOCK_SIZE, head_dim), dtype=cp.float32)
        self.block_tables = []
        self.seq_len = []
        self.free = list(range(num_blocks - 1, -1, -1))

    def new_sequence(self):
        self.block_tables.append([])
        self.seq_len.append(0)
        return len(self.block_tables) - 1

    def append(self, sid, pos, k, v):
        bi = pos // self.block_size
        while len(self.block_tables[sid]) <= bi:
            self.block_tables[sid].append(self.free.pop())
        pb = self.block_tables[sid][bi]
        slot = pos % self.block_size
        self.d_K[pb, slot] = cp.asarray(k, dtype=cp.float32)
        self.d_V[pb, slot] = cp.asarray(v, dtype=cp.float32)
        self.seq_len[sid] = max(self.seq_len[sid], pos + 1)

    def context(self, sid, qpos, Q):
        L = self.seq_len[sid]
        bt = cp.asarray(self.block_tables[sid], dtype=cp.int32)
        Kf = self.d_K.reshape(-1, self.head_dim)
        Vf = self.d_V.reshape(-1, self.head_dim)
        Ctx = cp.zeros((L, self.head_dim), dtype=cp.float32)
        _paged_ctx((1,), (1,), (Kf, Vf, bt, L, self.block_size,
                               self.head_dim, cp.asarray(Q, cp.float32),
                               Ctx, qpos), shared_mem=L * 4)
        return cp.asnumpy(Ctx[qpos])


def main():
    model = TinyTransformer()
    data = [
        [0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0],
        [1, 1, 2, 2, 3, 3, 4, 4], [0, 2, 4, 6, 0, 2, 4, 6],
    ]

    # gradient check
    seq = data[0]
    X, Q, K, V, a, C, H, L_ = model.forward(seq)
    base = model.loss(L_, seq)
    model.backward(seq, X, Q, K, V, a, C, H, L_)
    eps = 1e-3
    for w, gw, idx in [(model.Wq, model.gWq, (0, 0)),
                       (model.Wq, model.gWq, (1, 2)),
                       (model.Wc, model.gWc, (0, 0)),
                       (model.Wc, model.gWc, (1, 2))]:
        orig = w[idx]
        w[idx] = orig + eps
        _, _, _, _, _, _, _, Lp = model.forward(seq)
        w[idx] = orig - eps
        _, _, _, _, _, _, _, Lm = model.forward(seq)
        w[idx] = orig
        num = (model.loss(Lp, seq) - model.loss(Lm, seq)) / (2 * eps)
        print(f"  grad W{idx}: numeric={num:.5f} analytic={gw[idx]:.5f}")
    print(f"gradient check done (base loss {base:.4f})")

    # training
    print("\nTraining...")
    for epoch in range(200):
        total = 0.0
        for s in data:
            X, Q, K, V, a, C, H, L_ = model.forward(s)
            total += model.loss(L_, s)
            model.backward(s, X, Q, K, V, a, C, H, L_)
            model.sgd(0.02)  # conservative: this tiny unnormalized net diverges at higher lr
        if epoch % 40 == 0 or epoch == 199:
            print(f"  epoch {epoch:3d} avg loss {total / len(data):.4f}")

    # inference with paged KV cache
    print("\nInference with paged KV cache:")
    cache = PagedKVCache(num_blocks=64, head_dim=D)
    sid = cache.new_sequence()
    prompt = [0, 1, 2]
    generated = list(prompt)
    for t, tok in enumerate(prompt):
        k, v = model.token_kv(tok)
        cache.append(sid, t, k, v)
    for step in range(5):
        qpos = cache.seq_len[sid] - 1
        k_last, _ = model.token_kv(generated[-1])
        ctx = cache.context(sid, qpos, k_last)
        h = ctx @ model.Wo.T
        logits = h @ model.Wc          # (D,) @ (D,VOCAB) -> (VOCAB,)
        nxt = int(np.argmax(logits))
        k, v = model.token_kv(nxt)
        cache.append(sid, cache.seq_len[sid], k, v)
        generated.append(nxt)
        print(f"  step {step} -> token {nxt}")
    print("generated sequence:", generated)


if __name__ == "__main__":
    main()
