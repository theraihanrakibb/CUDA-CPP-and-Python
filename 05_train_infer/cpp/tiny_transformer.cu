// tiny_transformer.cu - train a tiny transformer, then serve it with a KV cache.
//
// Pipeline (the "train + infer with KV cache" story):
//   1. Train a minimal single-head attention language model on the CPU
//      (forward + manual backprop, with a finite-difference gradient check
//      so you can trust the gradients).
//   2. At inference, instead of recomputing every Key/Value each step, we
//      cache them in a PAGED KV cache (same idea as 04) and only compute the
//      new token's K/V, then read all cached K/V through the block table.
//
// Build:  make
// Run:    ./tiny_transformer
//
// NOTE: this file is self-contained (it re-includes the paged cache + the
// paged-attention kernel from project 04) so it compiles standalone.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                               \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(e));                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ----------------------------- paged KV cache -----------------------------
struct PagedKVCache {
    int block_size, num_blocks, num_heads, head_dim;
    float *d_K, *d_V;
    size_t block_bytes;
    std::vector<std::vector<int>> block_tables;
    std::vector<int> seq_len;
    std::vector<int> free_blocks;

    PagedKVCache(int bs, int nb, int nh, int hd)
        : block_size(bs), num_blocks(nb), num_heads(nh), head_dim(hd) {
        block_bytes = (size_t)block_size * nh * hd * sizeof(float);
        size_t total = (size_t)num_blocks * block_bytes;
        CUDA_CHECK(cudaMalloc(&d_K, total));
        CUDA_CHECK(cudaMalloc(&d_V, total));
        CUDA_CHECK(cudaMemset(d_K, 0, total));
        CUDA_CHECK(cudaMemset(d_V, 0, total));
        for (int i = nb - 1; i >= 0; --i) free_blocks.push_back(i);
    }
    ~PagedKVCache() { cudaFree(d_K); cudaFree(d_V); }
    int new_sequence() {
        int id = (int)block_tables.size();
        block_tables.push_back({});
        seq_len.push_back(0);
        return id;
    }
    int ensure_block(int seq, int pos) {
        int need = pos / block_size + 1;
        while ((int)block_tables[seq].size() < need) {
            if (free_blocks.empty()) { fprintf(stderr, "KV: out of blocks\n"); exit(1); }
            block_tables[seq].push_back(free_blocks.back());
            free_blocks.pop_back();
        }
        return block_tables[seq][pos / block_size];
    }
    void append(int seq, int pos, const float* k, const float* v) {
        int pb = ensure_block(seq, pos);
        int slot = pos % block_size;
        size_t off = (size_t)pb * block_bytes + (size_t)slot * num_heads * head_dim * sizeof(float);
        CUDA_CHECK(cudaMemcpy((char*)d_K + off, k, num_heads*head_dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((char*)d_V + off, v, num_heads*head_dim*sizeof(float), cudaMemcpyHostToDevice));
        if (pos + 1 > seq_len[seq]) seq_len[seq] = pos + 1;
    }
};

// One block per query position; thread 0 reads K/V via the block table and
// writes the attention CONTEXT c_t = sum_j a_tj * V_j  (this is the input to
// the output projection during inference).
__global__ void pagedAttentionContext(const float* d_K, const float* d_V,
                                      const int* block_table, int L, int block_size,
                                      int head_dim, const float* Q, float* Ctx, int qpos) {
    if (threadIdx.x != 0) return;
    extern __shared__ float scores[];
    float maxs = -1e30f;
    for (int j = 0; j < L; ++j) {
        int b = block_table[j / block_size];
        int slot = j % block_size;
        const float* k = d_K + ((size_t)b * block_size + slot) * head_dim;
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
        const float* v = d_V + ((size_t)b * block_size + slot) * head_dim;
        float w = scores[j] / sum;
        for (int d = 0; d < head_dim; ++d) Ctx[qpos * head_dim + d] += w * v[d];
    }
}

// ----------------------------- tiny model --------------------------------
struct Model {
    int d, vocab;
    std::vector<float> E, Wq, Wk, Wv, Wo, Wc;     // params
    std::vector<float> gE, gWq, gWk, gWv, gWo, gWc; // grads

    // activations (cleared each forward)
    std::vector<std::vector<float>> X, Q, K, V, C, H, LOGITS;
    std::vector<std::vector<float>> A;            // A[t] = attention weights up to t

    Model(int d_, int vocab_) : d(d_), vocab(vocab_) {
        auto rnd = [](int n, std::vector<float>& v) {
            v.resize(n);
            for (int i = 0; i < n; ++i) v[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        };
        rnd(vocab*d, E);  rnd(d*d, Wq);  rnd(d*d, Wk);
        rnd(d*d, Wv);     rnd(d*d, Wo);  rnd(d*vocab, Wc);
        gE.resize(vocab*d, 0); gWq.resize(d*d, 0); gWk.resize(d*d, 0);
        gWv.resize(d*d, 0); gWo.resize(d*d, 0); gWc.resize(d*vocab, 0);
    }

    static void matvec(const std::vector<float>& W, const std::vector<float>& x,
                       std::vector<float>& y, int nin, int nout) {
        for (int i = 0; i < nout; ++i) {
            float s = 0;
            for (int j = 0; j < nin; ++j) s += W[(size_t)i*nin + j] * x[j];
            y[i] = s;
        }
    }

    // forward over a token sequence; stores activations
    void forward(const std::vector<int>& seq) {
        int T = (int)seq.size();
        X.assign(T, std::vector<float>(d));
        Q.assign(T, std::vector<float>(d));
        K.assign(T, std::vector<float>(d));
        V.assign(T, std::vector<float>(d));
        C.assign(T, std::vector<float>(d));
        H.assign(T, std::vector<float>(d));
        LOGITS.assign(T, std::vector<float>(vocab));
        A.assign(T, std::vector<float>(0));

        float scale = 1.0f / sqrtf((float)d);
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < d; ++j) X[t][j] = E[(size_t)seq[t]*d + j];
            matvec(Wq, X[t], Q[t], d, d);
            matvec(Wk, X[t], K[t], d, d);
            matvec(Wv, X[t], V[t], d, d);
            // causal attention over u <= t
            A[t].resize(t + 1);
            float maxs = -1e30f;
            for (int u = 0; u <= t; ++u) {
                float s = 0;
                for (int j = 0; j < d; ++j) s += Q[t][j] * K[u][j];
                s *= scale;
                A[t][u] = s;
                if (s > maxs) maxs = s;
            }
            float sum = 0;
            for (int u = 0; u <= t; ++u) { A[t][u] = expf(A[t][u] - maxs); sum += A[t][u]; }
            for (int u = 0; u <= t; ++u) A[t][u] /= sum;
            std::fill(C[t].begin(), C[t].end(), 0);
            for (int u = 0; u <= t; ++u)
                for (int j = 0; j < d; ++j) C[t][j] += A[t][u] * V[u][j];
            matvec(Wo, C[t], H[t], d, d);
            matvec(Wc, H[t], LOGITS[t], d, vocab);
        }
    }

    // mean cross-entropy over next-token predictions t -> seq[t+1]
    float loss(const std::vector<int>& seq) {
        int T = (int)seq.size();
        int P = T - 1;
        float L = 0;
        for (int t = 0; t < P; ++t) {
            int target = seq[t + 1];
            float m = -1e30f;
            for (int i = 0; i < vocab; ++i) m = fmaxf(m, LOGITS[t][i]);
            float s = 0;
            for (int i = 0; i < vocab; ++i) s += expf(LOGITS[t][i] - m);
            L += logf(s) + m - LOGITS[t][target]; // = -log softmax(target)
        }
        return L / P;
    }

    // manual backprop; fills g* with dL/dparam (averaged over predictions)
    void backward(const std::vector<int>& seq) {
        int T = (int)seq.size();
        int P = T - 1;
        std::fill(gE.begin(), gE.end(), 0); std::fill(gWq.begin(), gWq.end(), 0);
        std::fill(gWk.begin(), gWk.end(), 0); std::fill(gWv.begin(), gWv.end(), 0);
        std::fill(gWo.begin(), gWo.end(), 0); std::fill(gWc.begin(), gWc.end(), 0);

        std::vector<std::vector<float>> glogits(T, std::vector<float>(vocab, 0));
        for (int t = 0; t < P; ++t) {
            int target = seq[t + 1];
            float m = -1e30f;
            for (int i = 0; i < vocab; ++i) m = fmaxf(m, LOGITS[t][i]);
            float s = 0;
            for (int i = 0; i < vocab; ++i) s += expf(LOGITS[t][i] - m);
            for (int i = 0; i < vocab; ++i) {
                float p = expf(LOGITS[t][i] - m) / s;
                glogits[t][i] = (p - (i == target ? 1.0f : 0.0f)) / P;
            }
        }
        float scale = 1.0f / sqrtf((float)d);
        for (int t = 0; t < T; ++t) {
            // output projection
            std::vector<float> gh(d, 0), gc(d, 0), gq(d, 0), gk(d, 0), gv(d, 0);
            for (int i = 0; i < vocab; ++i)
                for (int j = 0; j < d; ++j)
                    gWc[(size_t)i*d + j] += glogits[t][i] * H[t][j];
            for (int j = 0; j < d; ++j)
                for (int i = 0; i < vocab; ++i) gh[j] += Wc[(size_t)i*d + j] * glogits[t][i];
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j)
                    gWo[(size_t)i*d + j] += gh[i] * C[t][j];
            for (int j = 0; j < d; ++j)
                for (int i = 0; i < d; ++i) gc[j] += Wo[(size_t)i*d + j] * gh[i];
            // v gradient from context
            for (int u = 0; u <= t; ++u)
                for (int j = 0; j < d; ++j) gv[j] += A[t][u] * gc[j];
            // score gradients
            float gcdotc = 0;
            for (int j = 0; j < d; ++j) gcdotc += gc[j] * C[t][j];
            for (int u = 0; u <= t; ++u) {
                float gcdotv = 0;
                for (int j = 0; j < d; ++j) gcdotv += gc[j] * V[u][j];
                float gs = A[t][u] * (gcdotv - gcdotc);
                for (int j = 0; j < d; ++j) { gq[j] += gs * K[u][j] * scale; gk[j] += gs * Q[t][j] * scale; }
            }
            // input projections + embedding
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j) { gWq[(size_t)i*d + j] += gq[i]*X[t][j]; gWk[(size_t)i*d + j] += gk[i]*X[t][j]; gWv[(size_t)i*d + j] += gv[i]*X[t][j]; }
            std::vector<float> gx(d, 0);
            for (int j = 0; j < d; ++j) { gx[j] += gq[j] + gk[j] + gv[j]; }
            for (int j = 0; j < d; ++j) gE[(size_t)seq[t]*d + j] += gx[j];
        }
    }

    void sgd(float lr) {
        auto step = [&](std::vector<float>& p, std::vector<float>& g) {
            for (size_t i = 0; i < p.size(); ++i) p[i] -= lr * g[i];
        };
        step(E, gE); step(Wq, gWq); step(Wk, gWk); step(Wv, gWv); step(Wo, gWo); step(Wc, gWc);
    }

    // compute K for one token id (cpu) into out[d]
    void tokenK(int tok, std::vector<float>& k) {
        std::vector<float> x(d);
        for (int j = 0; j < d; ++j) x[j] = E[(size_t)tok*d + j];
        matvec(Wk, x, k, d, d);
    }
    void tokenV(int tok, std::vector<float>& v) {
        std::vector<float> x(d);
        for (int j = 0; j < d; ++j) x[j] = E[(size_t)tok*d + j];
        matvec(Wv, x, v, d, d);
    }
};

// ----------------------------- driver -------------------------------------
int main() {
    srand(1);
    const int d = 8, vocab = 12, T = 8;
    Model model(d, vocab);

    // tiny "dataset": a few short sequences
    std::vector<std::vector<int>> data = {
        {0,1,2,3,4,5,6,7}, {7,6,5,4,3,2,1,0}, {1,1,2,2,3,3,4,4}, {0,2,4,6,0,2,4,6}
    };

    // --- gradient check (finite differences) on a few weights ---
    {
        std::vector<int> s = data[0];
        model.forward(s);
        float base = model.loss(s);
        model.backward(s);
        float eps = 1e-3f;
        auto check = [&](const char* name, std::vector<float>& w, std::vector<float>& gw, int idx) {
            float old = w[idx];
            w[idx] = old + eps; model.forward(s); float lp = model.loss(s);
            w[idx] = old - eps; model.forward(s); float lm = model.loss(s);
            w[idx] = old; model.forward(s);
            float num = (lp - lm) / (2 * eps);
            float ana = gw[idx];
            printf("  grad %-4s[%d]: numeric=%.5f analytic=%.5f\n", name, idx, num, ana);
        };
        check("Wq", model.Wq, model.gWq, 0);
        check("Wq", model.Wq, model.gWq, 5);
        check("Wc", model.Wc, model.gWc, 0);
        check("Wc", model.Wc, model.gWc, 17);
        printf("gradient check done (base loss %.4f)\n", base);
    }

    // --- training ---
    printf("\nTraining...\n");
    for (int epoch = 0; epoch < 200; ++epoch) {
        float total = 0;
        for (auto& s : data) {
            model.forward(s);
            total += model.loss(s);
            model.backward(s);
            model.sgd(0.5f);
        }
        if (epoch % 40 == 0 || epoch == 199)
            printf("  epoch %3d  avg loss %.4f\n", epoch, total / data.size());
    }

    // --- inference WITH a paged KV cache on the GPU ---
    printf("\nInference with paged KV cache:\n");
    PagedKVCache cache(4, 64, 1, d);
    int seq = cache.new_sequence();
    std::vector<int> prompt = {0, 1, 2};
    std::vector<int> generated = prompt;
    for (int t = 0; t < (int)prompt.size(); ++t) {
        std::vector<float> k, v;
        model.tokenK(prompt[t], k);
        model.tokenV(prompt[t], v);
        cache.append(seq, t, k.data(), v.data());
    }
    int* d_bt;
    for (int step = 0; step < 5; ++step) {
        int L = cache.seq_len[seq];
        std::vector<int> bt = cache.block_tables[seq];
        CUDA_CHECK(cudaMalloc(&d_bt, bt.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_bt, bt.data(), bt.size() * sizeof(int), cudaMemcpyHostToDevice));

        int qpos = L - 1;
        std::vector<float> qk;
        model.tokenK(generated.back(), qk);             // query = last token's key
        float* d_Ctx; CUDA_CHECK(cudaMalloc(&d_Ctx, (size_t)L * d * sizeof(float)));
        pagedAttentionContext<<<1, 1, L * sizeof(float)>>>(
            cache.d_K, cache.d_V, d_bt, L, cache.block_size, d, qk.data(), d_Ctx, qpos);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> ctx(d);
        CUDA_CHECK(cudaMemcpy(ctx.data(), d_Ctx + (size_t)qpos*d, d*sizeof(float), cudaMemcpyDeviceToHost));

        // CPU read-out: h = Wo ctx ; logits = Wc h
        std::vector<float> h(d), logits(vocab);
        Model::matvec(model.Wo, ctx, h, d, d);
        Model::matvec(model.Wc, h, logits, d, vocab);
        int next = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());

        std::vector<float> nk, nv;
        model.tokenK(next, nk); model.tokenV(next, nv);
        cache.append(seq, L, nk.data(), nv.data());  // only the NEW token is cached
        generated.push_back(next);

        CUDA_CHECK(cudaFree(d_bt)); CUDA_CHECK(cudaFree(d_Ctx));
        printf("  step %d -> token %d\n", step, next);
    }
    printf("generated sequence: ");
    for (int t : generated) printf("%d ", t);
    printf("\n");
    return 0;
}
