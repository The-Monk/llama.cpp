#pragma once

// ROC8 cross-request prefix sharing (RadixAttention / APC for llama.cpp server).
//
// A token-level radix tree over the prompts of all *active* sequences. On a new
// request we find the longest prefix already resident in some other sequence's
// KV and seq_cp() those cells into the new slot (zero-copy, same-stream ->
// fp8-KV-safe), so only the divergent suffix is prefilled.
//
// The tree stores no KV itself. Each node tracks the set of seq_ids whose prompt
// passes through it; that set *is* the reference count. A prefix's KV is
// reclaimable exactly when the last referring sequence is released. Eviction is
// therefore driven by the server dropping sequences (LRU by last_used), not by
// the tree copying or owning data.
//
// Header-only + dependency-light so it can be unit-tested off the server:
// define ROC8_PREFIX_CACHE_TEST to typedef the llama ids locally.

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

#ifdef ROC8_PREFIX_CACHE_TEST
using llama_token   = int32_t;
using llama_seq_id  = int32_t;
#else
#include "llama.h"
#endif

struct server_prefix_match {
    int          len   = 0;   // number of leading tokens shareable from `owner`
    llama_seq_id owner = -1;  // a sequence whose KV covers [0, len); -1 = no match
};

class server_prefix_cache {
public:
    server_prefix_cache() : root(new node()) {}

    // Register that `seq`'s KV now covers exactly `tokens` (its full prompt).
    // Idempotent-ish: re-inserting the same seq refreshes last_used along its path.
    void insert(llama_seq_id seq, const std::vector<llama_token> & tokens) {
        tick++;
        node * cur = root.get();
        cur->seqs.insert(seq);
        cur->last_used = tick;
        size_t i = 0;
        while (i < tokens.size()) {
            const llama_token t = tokens[i];
            auto it = cur->children.find(t);
            if (it == cur->children.end()) {
                // fresh edge carrying the whole remaining suffix
                auto child = std::make_unique<node>();
                child->edge.assign(tokens.begin() + i, tokens.end());
                child->depth = cur->depth + (int) child->edge.size();
                child->seqs.insert(seq);
                child->last_used = tick;
                cur->children.emplace(t, std::move(child));
                return;
            }
            node * child = it->second.get();
            // match along the compressed edge
            size_t j = 0;
            while (j < child->edge.size() && i < tokens.size() && child->edge[j] == tokens[i]) {
                ++j; ++i;
            }
            if (j == child->edge.size()) {
                // consumed the whole edge -> descend
                child->seqs.insert(seq);
                child->last_used = tick;
                cur = child;
                continue;
            }
            // partial edge match -> split the edge at j.
            // Take ownership of the old child FIRST (child stays valid); only
            // replace the map entry at the very end.
            std::unique_ptr<node> oldchild = std::move(it->second);
            auto mid = std::make_unique<node>();
            mid->edge.assign(oldchild->edge.begin(), oldchild->edge.begin() + j);
            mid->depth = cur->depth + (int) mid->edge.size();
            mid->seqs = oldchild->seqs;         // everyone through oldchild also passes mid
            mid->seqs.insert(seq);
            mid->last_used = tick;
            node * midp = mid.get();
            // shorten the old child's edge and rewire it under mid
            oldchild->edge.erase(oldchild->edge.begin(), oldchild->edge.begin() + j);
            // oldchild->depth unchanged (still ends at the same absolute depth)
            const llama_token oldkey = oldchild->edge.front();
            midp->children.emplace(oldkey, std::move(oldchild));
            // remaining new suffix (if any) becomes a second child of mid
            if (i < tokens.size()) {
                auto tail = std::make_unique<node>();
                tail->edge.assign(tokens.begin() + i, tokens.end());
                tail->depth = midp->depth + (int) tail->edge.size();
                tail->seqs.insert(seq);
                tail->last_used = tick;
                midp->children.emplace(tokens[i], std::move(tail));
            }
            cur->children[t] = std::move(mid);   // finally replace the entry keyed by t
            return;
        }
    }

    // Longest prefix of `tokens` resident in some *other* sequence's KV.
    // Never returns `exclude` as the owner (don't share a sequence with itself).
    server_prefix_match match(const std::vector<llama_token> & tokens,
                              llama_seq_id exclude = -1) {
        tick++;
        server_prefix_match best;
        node * cur = root.get();
        size_t i = 0;
        while (i < tokens.size()) {
            auto it = cur->children.find(tokens[i]);
            if (it == cur->children.end()) break;
            node * child = it->second.get();
            size_t j = 0;
            while (j < child->edge.size() && i < tokens.size() && child->edge[j] == tokens[i]) {
                ++j; ++i;
            }
            const int matched = cur->depth + (int) j;
            const llama_seq_id owner = pick_owner(child, exclude);
            if (owner >= 0 && matched > best.len) {
                best.len   = matched;
                best.owner = owner;
            }
            child->last_used = tick;
            if (j < child->edge.size()) break;  // diverged mid-edge
            cur = child;                         // full edge -> keep descending
        }
        return best;
    }

    // Drop `seq` from every node on its prompt path; prune nodes with no
    // remaining referrers and no children.
    void release(llama_seq_id seq, const std::vector<llama_token> & tokens) {
        root->seqs.erase(seq);
        prune_path(root.get(), tokens, 0, seq);
    }

private:
    struct node {
        std::vector<llama_token> edge;
        std::unordered_map<llama_token, std::unique_ptr<node>> children;
        std::unordered_set<llama_seq_id> seqs;
        int      depth     = 0;
        uint64_t last_used = 0;
    };

    // Any referring seq != exclude works as an owner (its KV covers this prefix).
    static llama_seq_id pick_owner(node * n, llama_seq_id exclude) {
        for (llama_seq_id s : n->seqs) {
            if (s != exclude) return s;
        }
        return -1;
    }

    // recursive prune returning true if `child` node should be deleted by parent
    bool prune_path(node * cur, const std::vector<llama_token> & tokens, size_t i, llama_seq_id seq) {
        if (i >= tokens.size()) return false;
        auto it = cur->children.find(tokens[i]);
        if (it == cur->children.end()) return false;
        node * child = it->second.get();
        size_t j = 0;
        while (j < child->edge.size() && i < tokens.size() && child->edge[j] == tokens[i]) {
            ++j; ++i;
        }
        if (j != child->edge.size()) return false;  // path doesn't actually run through here
        child->seqs.erase(seq);
        prune_path(child, tokens, i, seq);
        if (child->seqs.empty() && child->children.empty()) {
            cur->children.erase(it);
        }
        return false;
    }

    std::unique_ptr<node> root;
    uint64_t tick = 0;
};
