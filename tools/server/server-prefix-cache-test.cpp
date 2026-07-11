// Standalone unit test for the ROC8 radix prefix cache.
//   g++ -std=c++17 -DROC8_PREFIX_CACHE_TEST server-prefix-cache-test.cpp -o /tmp/pc && /tmp/pc
#define ROC8_PREFIX_CACHE_TEST
#include "server-prefix-cache.h"
#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); ++fails; } else { printf("ok  : %s\n", msg); } } while (0)

using V = std::vector<llama_token>;

int main() {
    // system prompt shared by two conversations
    V sys = {1,2,3,4,5,6,7,8};
    V convA = sys; for (int t : {100,101,102}) convA.push_back(t);   // sys + A
    V convB = sys; for (int t : {200,201})     convB.push_back(t);   // sys + B

    // --- basic cross-request share ---
    {
        server_prefix_cache pc;
        pc.insert(0, convA);                 // seq 0 = sys + A resident
        auto m = pc.match(convB, /*exclude*/1);
        CHECK(m.len == (int) sys.size() && m.owner == 0, "convB shares full system prefix from seq0");
        CHECK(m.len < (int) convB.size(), "shared prefix is shorter than full convB (suffix still prefills)");
    }

    // --- exclude self: a seq must not match itself ---
    {
        server_prefix_cache pc;
        pc.insert(0, convA);
        auto m = pc.match(convA, /*exclude*/0);
        CHECK(m.owner != 0, "match excludes the querying seq itself");
        CHECK(m.len == 0, "no other seq -> no shareable prefix");
    }

    // --- exact full-length match from another seq ---
    {
        server_prefix_cache pc;
        pc.insert(0, convA);
        auto m = pc.match(convA, /*exclude*/7);
        CHECK(m.len == (int) convA.size() && m.owner == 0, "identical prompt shares entire length");
    }

    // --- no overlap ---
    {
        server_prefix_cache pc;
        pc.insert(0, {9,9,9,9});
        auto m = pc.match({8,8,8}, -1);
        CHECK(m.len == 0 && m.owner == -1, "disjoint prompt -> no match");
    }

    // --- edge split: insert long, then a prompt diverging mid-edge ---
    {
        server_prefix_cache pc;
        pc.insert(0, {1,2,3,4,5});
        // diverges after {1,2,3}
        auto m = pc.match({1,2,3,7,7}, -1);
        CHECK(m.len == 3 && m.owner == 0, "divergence mid-edge returns matched length 3");
        // now insert the diverging one and re-match the original
        pc.insert(1, {1,2,3,7,7});
        auto m2 = pc.match({1,2,3,4,5}, /*exclude*/0);
        CHECK(m2.len == 3 && m2.owner == 1, "after split, {1,2,3} shareable from the other branch");
    }

    // --- refcount / release: two seqs share sys, release one, still shareable ---
    {
        server_prefix_cache pc;
        pc.insert(0, convA);
        pc.insert(1, convB);
        auto m = pc.match(sys, /*exclude*/2);
        CHECK(m.len == (int) sys.size(), "third request shares system prefix (2 owners present)");
        pc.release(0, convA);                // seq0 gone
        auto m2 = pc.match(sys, /*exclude*/2);
        CHECK(m2.len == (int) sys.size() && m2.owner == 1, "prefix still shareable from remaining seq1");
        pc.release(1, convB);                // last owner gone
        auto m3 = pc.match(sys, /*exclude*/2);
        CHECK(m3.len == 0, "prefix reclaimed once last referrer released");
    }

    // --- release prunes only the vanished branch, keeps shared trunk ---
    {
        server_prefix_cache pc;
        pc.insert(0, convA);   // sys + A
        pc.insert(1, convB);   // sys + B
        pc.release(0, convA);  // drop A branch
        auto m = pc.match(convB, /*exclude*/9);
        CHECK(m.len == (int) convB.size() && m.owner == 1, "seq1 fully intact after seq0 pruned");
    }

    printf(fails ? "\nTESTS FAILED (%d)\n" : "\nALL PASS\n", fails);
    return fails ? 1 : 0;
}
