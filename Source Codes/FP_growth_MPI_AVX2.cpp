#define _CRT_SECURE_NO_WARNINGS
#include <mpi.h>
#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>

// ============================================================
// AVX2 helpers (bulk int copy)
// ============================================================

// Copy n int32 from src to dst using AVX2 (8 ints per iteration)
static inline void avx2_copy_i32(const int* src, int* dst, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
    for (; i < n; i++) dst[i] = src[i];
}

// Append prefix ints into a vector<int> using AVX2 copy
static inline void avx2_append_prefix(std::vector<int>& out, const std::vector<int>& prefix) {
    if (prefix.empty()) return;
    size_t old = out.size();
    out.resize(old + prefix.size());
    avx2_copy_i32(prefix.data(), out.data() + old, (int)prefix.size());
}

// ============================================================
// Basic utilities
// ============================================================

static int64_t count_file_lines(const char* file_name) {
    std::ifstream f(file_name);
    if (!f.is_open()) return -1;
    int64_t lines = 0;
    std::string line;
    while (std::getline(f, line)) lines++;
    return lines;
}

static void compute_local_start_end(int64_t total_lines, int my_rank, int comm_sz,
                                    int64_t* local_start, int64_t* local_end) {
    int64_t base = total_lines / comm_sz;
    int64_t rem  = total_lines % comm_sz;
    int64_t my_lines = base + (my_rank < rem ? 1 : 0);

    int64_t start = 0;
    for (int r = 0; r < my_rank; r++) start += base + (r < rem ? 1 : 0);

    *local_start = start;
    *local_end   = start + my_lines;
}

// ============================================================
// FP-tree
// ============================================================

struct FPNode {
    int item;              // item id
    int64_t count;
    FPNode* parent;
    std::unordered_map<int, FPNode*> children;
    FPNode* next;          // header link

    FPNode(int it, int64_t c, FPNode* p) : item(it), count(c), parent(p), next(nullptr) {}
};

struct FPTree {
    FPNode* root;
    std::vector<FPNode*> header;
    std::vector<int64_t> item_count;

    explicit FPTree(int num_items)
        : root(new FPNode(-1, 0, nullptr)),
          header(num_items, nullptr),
          item_count(num_items, 0) {}

    ~FPTree() { free_all(root); }

    static void free_all(FPNode* node) {
        if (!node) return;
        for (auto& kv : node->children) free_all(kv.second);
        delete node;
    }

    void add_transaction(const std::vector<int>& items, int64_t cnt) {
        FPNode* cur = root;
        for (int it : items) {
            auto found = cur->children.find(it);
            if (found == cur->children.end()) {
                FPNode* child = new FPNode(it, cnt, cur);
                cur->children[it] = child;
                child->next = header[it];
                header[it] = child;
                cur = child;
            } else {
                found->second->count += cnt;
                cur = found->second;
            }
        }
    }

    bool is_single_path() const {
        FPNode* cur = root;
        while (cur) {
            if (cur->children.empty()) return true;
            if (cur->children.size() > 1) return false;
            cur = cur->children.begin()->second;
        }
        return true;
    }
};

static void build_conditional_pattern_base(const FPTree& tree, int base_item,
    std::vector<std::pair<std::vector<int>, int64_t>>& out_paths)
{
    out_paths.clear();
    FPNode* node = tree.header[base_item];
    while (node) {
        int64_t cnt = node->count;
        std::vector<int> prefix;

        FPNode* p = node->parent;
        while (p && p->item != -1) {
            prefix.push_back(p->item);
            p = p->parent;
        }
        std::reverse(prefix.begin(), prefix.end());
        if (!prefix.empty()) out_paths.push_back({ std::move(prefix), cnt });
        node = node->next;
    }
}

static FPTree* build_conditional_tree(
    const std::vector<std::pair<std::vector<int>, int64_t>>& paths,
    int num_items, int64_t min_count,
    const std::vector<int>& global_rank_order)
{
    std::vector<int64_t> freq(num_items, 0);
    for (const auto& pc : paths) {
        const auto& path = pc.first;
        int64_t cnt = pc.second;
        for (int it : path) freq[it] += cnt;
    }

    bool any = false;
    for (int it = 0; it < num_items; it++) if (freq[it] >= min_count) { any = true; break; }
    if (!any) return nullptr;

    auto cmp = [&](int a, int b) {
        if (freq[a] != freq[b]) return freq[a] > freq[b];
        return global_rank_order[a] < global_rank_order[b];
    };

    FPTree* ct = new FPTree(num_items);
    ct->item_count = freq;

    std::vector<int> filtered;
    for (const auto& pc : paths) {
        filtered.clear();
        const auto& path = pc.first;
        int64_t cnt = pc.second;

        for (int it : path) if (freq[it] >= min_count) filtered.push_back(it);
        if (filtered.empty()) continue;

        std::sort(filtered.begin(), filtered.end(), cmp);
        ct->add_transaction(filtered, cnt);
    }

    if (ct->root->children.empty()) {
        delete ct;
        return nullptr;
    }
    return ct;
}

static void fp_growth_mine(
    const FPTree& tree, int64_t min_count,
    std::vector<int>& suffix,
    std::vector<std::vector<int>>& out_patterns,
    const std::vector<int>& global_rank_order)
{
    if (tree.is_single_path()) {
        std::vector<std::pair<int, int64_t>> path;
        FPNode* cur = tree.root;
        while (cur && !cur->children.empty()) {
            cur = cur->children.begin()->second;
            path.push_back({ cur->item, cur->count });
        }

        const int m = (int)path.size();
        if (m == 0) return;

        // enumerate all non-empty subsets
        if (m <= 30) {
            for (int mask = 1; mask < (1 << m); mask++) {
                std::vector<int> pat = suffix;
                for (int i = 0; i < m; i++) if (mask & (1 << i)) pat.push_back(path[i].first);
                out_patterns.push_back(std::move(pat));
            }
        } else {
            // too long; avoid overflow
            // (can implement iterative combination generation if needed)
        }
        return;
    }

    std::vector<int> items;
    items.reserve((int)tree.header.size());
    for (int it = 0; it < (int)tree.header.size(); it++) {
        if (tree.item_count[it] >= min_count && tree.header[it] != nullptr) items.push_back(it);
    }

    std::sort(items.begin(), items.end(), [&](int a, int b) {
        if (tree.item_count[a] != tree.item_count[b]) return tree.item_count[a] < tree.item_count[b];
        return global_rank_order[a] > global_rank_order[b];
    });

    for (int base_item : items) {
        suffix.push_back(base_item);
        out_patterns.push_back(suffix);

        std::vector<std::pair<std::vector<int>, int64_t>> cpb;
        build_conditional_pattern_base(tree, base_item, cpb);

        FPTree* ct = build_conditional_tree(cpb, (int)tree.header.size(), min_count, global_rank_order);
        if (ct) {
            fp_growth_mine(*ct, min_count, suffix, out_patterns, global_rank_order);
            delete ct;
        }
        suffix.pop_back();
    }
}

// ============================================================
// MPI projected DB (PFP-style)
// owner(item) = item % comm_sz
// ============================================================

static inline int owner_of_item(int item_id, int comm_sz) { return item_id % comm_sz; }

// Pack: [baseItem, lenPrefix, count, prefix..., baseItem, lenPrefix, count, prefix...]
static inline void pack_prefix(std::vector<int>& buf, int baseItem, const std::vector<int>& prefix, int count) {
    buf.push_back(baseItem);
    buf.push_back((int)prefix.size());
    buf.push_back(count);
    avx2_append_prefix(buf, prefix);  // AVX2 bulk copy of prefix ints
}

static void unpack_received(
    const std::vector<int>& buf,
    std::vector<std::vector<std::pair<std::vector<int>, int64_t>>>& projected_db)
{
    size_t i = 0;
    while (i + 3 <= buf.size()) {
        int baseItem = buf[i++];
        int len      = buf[i++];
        int cnt      = buf[i++];

        if (i + (size_t)len > buf.size()) break;

        std::vector<int> prefix(len);
        // AVX2 copy from buffer into prefix
        avx2_copy_i32(buf.data() + (int)i, prefix.data(), len);
        i += (size_t)len;

        if (!prefix.empty()) projected_db[baseItem].push_back({ std::move(prefix), (int64_t)cnt });
    }
}

// ============================================================
// Reading & global frequency
// We'll keep tokens as strings only for the FIRST global pass.
// Then we map to IDs and everything else is int-based.
// ============================================================

// Serialize local string counts as joined keys + parallel counts
static void serialize_string_counts(const std::unordered_map<std::string, int64_t>& m,
                                   std::string& keys_joined,
                                   std::vector<int64_t>& vals) {
    keys_joined.clear();
    vals.clear();
    keys_joined.reserve(4096);
    vals.reserve(m.size());
    bool first = true;
    for (const auto& kv : m) {
        if (!first) keys_joined.push_back('|');
        first = false;
        keys_joined += kv.first;
        vals.push_back(kv.second);
    }
}

static void deserialize_add_string_counts(const std::string& keys_joined,
                                         const std::vector<int64_t>& vals,
                                         std::unordered_map<std::string, int64_t>& dst) {
    if (keys_joined.empty()) return;
    std::stringstream ss(keys_joined);
    std::string k;
    int idx = 0;
    while (std::getline(ss, k, '|')) {
        if (k.empty()) continue;
        if (idx < (int)vals.size()) dst[k] += vals[idx];
        idx++;
    }
}

// Broadcast vector<string> with '|'
static void bcast_string_vector(std::vector<std::string>& v, int my_rank) {
    std::string joined;
    int count = 0;
    if (my_rank == 0) {
        for (size_t i = 0; i < v.size(); i++) {
            if (!joined.empty()) joined.push_back('|');
            joined += v[i];
        }
        count = (int)joined.size() + 1;
    }
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<char> buf(count);
    if (my_rank == 0 && count > 0) std::memcpy(buf.data(), joined.c_str(), count);
    MPI_Bcast(buf.data(), count, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (my_rank != 0) {
        v.clear();
        if (count > 1) {
            std::string recv(buf.data());
            std::stringstream ss(recv);
            std::string item;
            while (std::getline(ss, item, '|')) if (!item.empty()) v.push_back(item);
        }
    }
}

static void split_ws(const std::string& line, std::vector<std::string>& out) {
    out.clear();
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) out.push_back(tok);
}

static void read_partition_and_count_strings(
    const char* file_name,
    int64_t local_start, int64_t local_end,
    std::vector<std::vector<std::string>>& local_tx,
    std::unordered_map<std::string, int64_t>& local_counts)
{
    std::ifstream f(file_name);
    if (!f.is_open()) {
        std::cerr << "Unable to open file: " << file_name << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    local_tx.clear();
    local_counts.clear();

    std::string line;
    int64_t line_no = 0;
    std::vector<std::string> toks;

    while (std::getline(f, line)) {
        if (line_no >= local_end) break;
        if (line_no >= local_start && line_no < local_end) {
            split_ws(line, toks);
            if (!toks.empty()) {
                local_tx.push_back(toks);
                for (const auto& t : toks) local_counts[t] += 1;
            }
        }
        line_no++;
    }
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv) {
        auto start = std::chrono::high_resolution_clock::now();

    MPI_Init(&argc, &argv);
    double t_start = MPI_Wtime();

    int my_rank = 0, comm_sz = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc < 3) {
        if (my_rank == 0) {
            std::cerr << "Usage: mpiexec -n N fp_growth_mpi_avx2.exe <input_file> <min_support>\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const char* file_name = argv[1];
    double min_support = std::atof(argv[2]);
    if (!(min_support > 0.0 && min_support <= 1.0)) {
        if (my_rank == 0) std::cerr << "min_support must be in (0, 1].\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int64_t total_lines = 0;
    if (my_rank == 0) {
        total_lines = count_file_lines(file_name);
        if (total_lines < 0) {
            std::cerr << "Unable to open file: " << file_name << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&total_lines, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    int64_t local_start = 0, local_end = 0;
    compute_local_start_end(total_lines, my_rank, comm_sz, &local_start, &local_end);

    // -----------------------------
    // Pass 1: local string counts
    // -----------------------------
    std::vector<std::vector<std::string>> local_tx_str;
    std::unordered_map<std::string, int64_t> local_counts_str;
    read_partition_and_count_strings(file_name, local_start, local_end, local_tx_str, local_counts_str);

    // -----------------------------
    // Gather global counts on rank0
    // -----------------------------
    std::string keys_joined;
    std::vector<int64_t> vals;
    serialize_string_counts(local_counts_str, keys_joined, vals);

    std::unordered_map<std::string, int64_t> global_counts_str;

    if (my_rank == 0) {
        global_counts_str = local_counts_str;

        for (int src = 1; src < comm_sz; src++) {
            int len = 0, n = 0;
            MPI_Recv(&len, 1, MPI_INT, src, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&n,   1, MPI_INT, src, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::string recv_keys;
            std::vector<int64_t> recv_vals;

            if (len > 1) {
                std::vector<char> buf(len);
                MPI_Recv(buf.data(), len, MPI_CHAR, src, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recv_keys = std::string(buf.data());
            }
            if (n > 0) {
                recv_vals.resize(n);
                MPI_Recv(recv_vals.data(), n, MPI_LONG_LONG, src, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            deserialize_add_string_counts(recv_keys, recv_vals, global_counts_str);
        }
    } else {
        int len = (int)keys_joined.size() + 1;
        int n   = (int)vals.size();
        MPI_Send(&len, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
        MPI_Send(&n,   1, MPI_INT, 0, 11, MPI_COMM_WORLD);
        if (len > 1) MPI_Send(keys_joined.c_str(), len, MPI_CHAR, 0, 12, MPI_COMM_WORLD);
        if (n > 0)   MPI_Send(vals.data(), n, MPI_LONG_LONG, 0, 13, MPI_COMM_WORLD);
    }

    int64_t min_count = (int64_t)std::ceil(min_support * (double)total_lines);
    if (min_count < 1) min_count = 1;

    // -----------------------------
    // Rank0: build frequent list ordered by descending count
    // -----------------------------
    std::vector<std::string> freq_items_ordered;
    std::vector<int64_t> freq_counts_ordered;

    if (my_rank == 0) {
        std::vector<std::pair<std::string, int64_t>> items;
        items.reserve(global_counts_str.size());
        for (const auto& kv : global_counts_str) {
            if (kv.second >= min_count) items.push_back(kv);
        }
        std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });

        freq_items_ordered.reserve(items.size());
        freq_counts_ordered.reserve(items.size());
        for (const auto& kv : items) {
            freq_items_ordered.push_back(kv.first);
            freq_counts_ordered.push_back(kv.second);
        }
    }

    // broadcast frequent item names in global order
    bcast_string_vector(freq_items_ordered, my_rank);

    int num_items = (int)freq_items_ordered.size();
    MPI_Bcast(&num_items, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // broadcast counts too (optional)
    std::vector<long long> counts_ll(num_items);
    if (my_rank == 0) {
        for (int i = 0; i < num_items; i++) counts_ll[i] = (long long)freq_counts_ordered[i];
    }
    if (num_items > 0) MPI_Bcast(counts_ll.data(), num_items, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    if (my_rank != 0) {
        freq_counts_ordered.assign(num_items, 0);
        for (int i = 0; i < num_items; i++) freq_counts_ordered[i] = (int64_t)counts_ll[i];
    }

    // item->id map
    std::unordered_map<std::string, int> item_to_id;
    item_to_id.reserve((size_t)num_items * 2);
    for (int i = 0; i < num_items; i++) item_to_id[freq_items_ordered[i]] = i;

    // global order rank = id (0 is most frequent)
    std::vector<int> global_rank_order(num_items);
    for (int i = 0; i < num_items; i++) global_rank_order[i] = i;

    // -----------------------------
    // Convert local transactions to int IDs + sort + unique
    // -----------------------------
    std::vector<std::vector<int>> local_tx_ids;
    local_tx_ids.reserve(local_tx_str.size());

    for (const auto& tx : local_tx_str) {
        std::vector<int> ids;
        ids.reserve(tx.size());
        for (const auto& s : tx) {
            auto it = item_to_id.find(s);
            if (it != item_to_id.end()) ids.push_back(it->second);
        }
        if (ids.empty()) continue;
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        if (!ids.empty()) local_tx_ids.push_back(std::move(ids));
    }

    // -----------------------------
    // Build projected DB and redistribute with MPI_Alltoallv
    // AVX2 used heavily in packing/unpacking prefixes
    // -----------------------------
    std::vector<std::vector<int>> sendbufs(comm_sz);
    std::vector<int> prefix;

    for (const auto& tx : local_tx_ids) {
        prefix.clear();
        for (int i = 0; i < (int)tx.size(); i++) {
            int baseItem = tx[i];
            int dest = owner_of_item(baseItem, comm_sz);
            prefix.assign(tx.begin(), tx.begin() + i);
            pack_prefix(sendbufs[dest], baseItem, prefix, 1);
        }
    }

    std::vector<int> sendcounts(comm_sz, 0), recvcounts(comm_sz, 0);
    for (int r = 0; r < comm_sz; r++) sendcounts[r] = (int)sendbufs[r].size();

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> sdispls(comm_sz, 0), rdispls(comm_sz, 0);
    int sTotal = 0, rTotal = 0;
    for (int r = 0; r < comm_sz; r++) { sdispls[r] = sTotal; sTotal += sendcounts[r]; }
    for (int r = 0; r < comm_sz; r++) { rdispls[r] = rTotal; rTotal += recvcounts[r]; }

    std::vector<int> sendflat(sTotal);
    for (int r = 0; r < comm_sz; r++) {
        if (!sendbufs[r].empty()) {
            avx2_copy_i32(sendbufs[r].data(), sendflat.data() + sdispls[r], (int)sendbufs[r].size());
        }
    }

    std::vector<int> recvflat(rTotal);

    MPI_Alltoallv(sendflat.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                  recvflat.data(), recvcounts.data(), rdispls.data(), MPI_INT,
                  MPI_COMM_WORLD);

    std::vector<std::vector<std::pair<std::vector<int>, int64_t>>> projected_db(num_items);
    unpack_received(recvflat, projected_db);

    // -----------------------------
    // Mine locally owned items
    // -----------------------------
    std::vector<std::vector<int>> local_patterns;
    local_patterns.reserve(4096);

    for (int baseItem = 0; baseItem < num_items; baseItem++) {
        if (owner_of_item(baseItem, comm_sz) != my_rank) continue;
        const auto& paths = projected_db[baseItem];
        if (paths.empty()) continue;

        FPTree* ct = build_conditional_tree(paths, num_items, min_count, global_rank_order);
        if (!ct) continue;

        std::vector<int> suffix;
        suffix.push_back(baseItem);
        local_patterns.push_back(suffix);

        fp_growth_mine(*ct, min_count, suffix, local_patterns, global_rank_order);

        delete ct;
    }

    int local_pat_count = (int)local_patterns.size();
    int global_pat_count = 0;
    MPI_Reduce(&local_pat_count, &global_pat_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();

    if (my_rank == 0) {
        std::cout << "FP-Growth (MPI + AVX2 bulk packing/unpacking) finished.\n";
        std::cout << "Transactions: " << total_lines << "\n";
        std::cout << "Min support: " << min_support << "  => min_count = " << min_count << "\n";
        std::cout << "Frequent items: " << num_items << "\n";
        std::cout << "Total mined patterns (approx, may include duplicates in some configs): " << global_pat_count << "\n";
        std::cout << "\nTotal execution time: " << (t_end - t_start) << " seconds\n";
    }

    MPI_Finalize();
        auto end = std::chrono::high_resolution_clock::now();

    auto duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(end-start);

    std::cout<<"\nExecution Time: "
             <<duration.count()<<" ms\n";
    return 0;
}
