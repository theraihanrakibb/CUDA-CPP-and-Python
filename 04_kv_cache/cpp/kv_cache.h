// kv_cache.h - a tiny PAGED KV cache manager (CPU side).
//
// The manager owns one big contiguous GPU buffer and hands out fixed-size
// "blocks" (like pages in virtual memory). Each sequence maps its logical
// token positions to physical blocks through a per-sequence block table.
// This is the same idea as vLLM's PagedAttention, scaled down to teach.
//
// Only the *storage* lives here; the GPU attention math is in
// paged_attention.cu. The manager just decides WHERE each token's K/V go.

#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

struct PagedKVCache {
    int block_size;   // tokens per physical block
    int num_blocks;   // total physical blocks
    int num_heads;
    int head_dim;
    float* d_K;       // physical K: [num_blocks][block_size][num_heads][head_dim]
    float* d_V;       // physical V: same shape
    size_t block_bytes;                       // bytes for one (block, head) slice
    std::vector<std::vector<int>> block_tables; // seq -> list of physical block ids
    std::vector<int> seq_len;                 // current length per sequence
    std::vector<int> free_blocks;             // physical ids not yet handed out

    PagedKVCache(int block_size_, int num_blocks_, int num_heads_, int head_dim_)
        : block_size(block_size_), num_blocks(num_blocks_),
          num_heads(num_heads_), head_dim(head_dim_) {
        block_bytes = (size_t)block_size * num_heads * head_dim * sizeof(float);
        size_t total = (size_t)num_blocks * block_bytes;
        cudaMalloc(&d_K, total);
        cudaMalloc(&d_V, total);
        cudaMemset(d_K, 0, total);
        cudaMemset(d_V, 0, total);
        for (int i = num_blocks - 1; i >= 0; --i) free_blocks.push_back(i);
    }

    ~PagedKVCache() { cudaFree(d_K); cudaFree(d_V); }

    // Register a new sequence; returns its id.
    int new_sequence() {
        int id = (int)block_tables.size();
        block_tables.push_back({});
        seq_len.push_back(0);
        return id;
    }

    // Make sure logical position 'pos' of 'seq' has a physical block allocated.
    int ensure_block(int seq, int pos) {
        int need = pos / block_size + 1;
        while ((int)block_tables[seq].size() < need) {
            if (free_blocks.empty()) {
                fprintf(stderr, "PagedKVCache: out of physical blocks\n");
                exit(EXIT_FAILURE);
            }
            block_tables[seq].push_back(free_blocks.back());
            free_blocks.pop_back();
        }
        return block_tables[seq][pos / block_size];
    }

    // Append one token's K and V (each length num_heads*head_dim) at (seq, pos).
    void append(int seq, int pos, const float* k, const float* v) {
        int pb = ensure_block(seq, pos);
        int slot = pos % block_size;
        size_t off = (size_t)pb * block_bytes + (size_t)slot * num_heads * head_dim * sizeof(float);
        cudaMemcpy((char*)d_K + off, k, num_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((char*)d_V + off, v, num_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice);
        if (pos + 1 > seq_len[seq]) seq_len[seq] = pos + 1;
    }

    // Helper for tests: pull a sequence's K/V into dense host arrays [L][num_heads*head_dim].
    void dump_dense(int seq, float* outK, float* outV) const {
        int L = seq_len[seq];
        int stride = num_heads * head_dim;
        for (int pos = 0; pos < L; ++pos) {
            int pb = block_tables[seq][pos / block_size];
            int slot = pos % block_size;
            size_t off = (size_t)pb * block_bytes + (size_t)slot * stride * sizeof(float);
            cudaMemcpy(outK + (size_t)pos * stride, (char*)d_K + off, stride * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(outV + (size_t)pos * stride, (char*)d_V + off, stride * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
};

#endif // KV_CACHE_H
