#define _CRT_SECURE_NO_WARNINGS
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
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

static inline void avx2_copy_i32(const int* src, int* dst, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
    for (; i < n; i++) dst[i] = src[i];
}

static inline void avx2_append(std::vector<int>& out, const int* src, int n) {
    if (n <= 0) return;
    size_t old = out.size();
    out.resize(old + (size_t)n);
    avx2_copy_i32(src, out.data() + old, n);
}

// ============================================================
// Utility
// ============================================================

static int64_t count_file_lines(const char* file_name) {
    std::ifstream f(file_name);
    if (!f.is_open()) return -1;
    int64_t lines = 0;
    std::string line;
    while (std::getline(f, line)) lines++;
    return lines;
}

static void split_ws(const std::string& line, std::vector<std::string>& out) {
    out.clear();
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) out.push_back(tok);
}

// ============================================================
// FP-tree structures (single-threaded build, read-only mine)
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

// Build conditional pattern base for base_item: list of (prefix, count)
static void build_conditional_pattern_base(const FPTree& tree, int base_item,
                                           std::vector<std::pair<std::vector<int>, int64_t>>& out_paths) {
    out_paths.clear();
    FPNode* node = tree.header[base_item];
    while (node) {
        int64_t cnt = node->count;

        // gather prefix path
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
    for (int it = 0; it < num_items; it++) {
        if (freq[it] >= min_count) { any = true; break; }
    }
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

        filtered.reserve(path.size());
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
    int64_t& out_pattern_count,
    const std::vector<int>& global_rank_order)
{
    // count patterns only (printing everything is usually huge)

    if (tree.is_single_path()) {
        // enumerate subsets
        std::vector<int> path_items;
        FPNode* cur = tree.root;
        while (cur && !cur->children.empty()) {
            cur = cur->children.begin()->second;
            path_items.push_back(cur->item);
        }
        int m = (int)path_items.size();
        if (m == 0) return;

        if (m <= 30) {
            // number of non-empty subsets = 2^m - 1
            out_pattern_count += ((int64_t)1 << m) - 1;
        } else {
            // avoid overflow; still count approximately
            out_pattern_count += (int64_t)0; // or handle combinatorics safely if needed
        }
        return;
    }

    // items present in this conditional tree
    std::vector<int> items;
    items.reserve((int)tree.header.size());
    for (int it = 0; it < (int)tree.header.size(); it++) {
        if (tree.item_count[it] >= min_count && tree.header[it] != nullptr) items.push_back(it);
    }

    // process in increasing count order
    std::sort(items.begin(), items.end(), [&](int a, int b) {
        if (tree.item_count[a] != tree.item_count[b]) return tree.item_count[a] < tree.item_count[b];
        return global_rank_order[a] > global_rank_order[b];
    });

    for (int base_item : items) {
        suffix.push_back(base_item);

        // the pattern "suffix" itself is frequent
        out_pattern_count += 1;

        std::vector<std::pair<std::vector<int>, int64_t>> cpb;
        build_conditional_pattern_base(tree, base_item, cpb);

        FPTree* ct = build_conditional_tree(cpb, (int)tree.header.size(), min_count, global_rank_order);
        if (ct) {
            fp_growth_mine(*ct, min_count, suffix, out_pattern_count, global_rank_order);
            delete ct;
        }

        suffix.pop_back();
    }
}

// ============================================================
// Build projected DB (prefix lists) using AVX2 bulk copies
// For each transaction: for each item position i, we store a record for base=item[i]:
//   record = [lenPrefix, count(=1), prefixItems...]
// We store per base item in one vector<int> (flat).
// ============================================================

static inline void append_record_avx(std::vector<int>& out, const int* prefix, int len, int count) {
    out.push_back(len);
    out.push_back(count);
    avx2_append(out, prefix, len);
}

static void build_projected_db_avx(
    const std::vector<std::vector<int>>& transactions,
    int num_items,
    std::vector<std::vector<int>>& projected_flat)
{
    projected_flat.assign(num_items, {});
    // Reserve a bit to reduce reallocs (rough guess)
    for (int i = 0; i < num_items; i++) projected_flat[i].reserve(1024);

    for (const auto& tx : transactions) {
        const int n = (int)tx.size();
        // tx is ordered (most frequent first => lower id first)
        for (int i = 0; i < n; i++) {
            int base = tx[i];
            int lenPrefix = i;
            if (lenPrefix == 0) continue; // empty prefix -> no conditional base contribution
            append_record_avx(projected_flat[base], tx.data(), lenPrefix, 1);
        }
    }
}

// Convert projected_flat[base] into vector of (prefix, count)
static void unpack_projected_db(
    const std::vector<int>& flat,
    std::vector<std::pair<std::vector<int>, int64_t>>& out_paths)
{
    out_paths.clear();
    size_t i = 0;
    while (i + 2 <= flat.size()) {
        int len = flat[i++];
        int cnt = flat[i++];
        if (len <= 0) continue;
        if (i + (size_t)len > flat.size()) break;

        std::vector<int> prefix(len);
        avx2_copy_i32(flat.data() + (int)i, prefix.data(), len);
        i += (size_t)len;

        out_paths.push_back({ std::move(prefix), (int64_t)cnt });
    }
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv) {
            auto start = std::chrono::high_resolution_clock::now();

    if (argc < 3) {
        std::cerr << "Usage: fp_growth_omp_avx2.exe <input_file> <min_support>\n";
        std::cerr << "Example: fp_growth_omp_avx2.exe data.txt 0.05\n";
        return 1;
    }

    const char* file_name = argv[1];
    double min_support = std::atof(argv[2]);
    if (!(min_support > 0.0 && min_support <= 1.0)) {
        std::cerr << "min_support must be in (0,1].\n";
        return 1;
    }

    double t_start = omp_get_wtime();

    int64_t total_lines = count_file_lines(file_name);
    if (total_lines < 0) {
        std::cerr << "Unable to open file: " << file_name << "\n";
        return 1;
    }

    int64_t min_count = (int64_t)std::ceil(min_support * (double)total_lines);
    if (min_count < 1) min_count = 1;

    // ------------------------------------------------------------
    // Pass 1: read full file, count items globally (single process)
    // ------------------------------------------------------------
    std::ifstream f(file_name);
    if (!f.is_open()) {
        std::cerr << "Unable to open file: " << file_name << "\n";
        return 1;
    }

    std::unordered_map<std::string, int64_t> counts;
    std::vector<std::vector<std::string>> transactions_str;
    transactions_str.reserve((size_t)total_lines);

    std::string line;
    std::vector<std::string> toks;

    while (std::getline(f, line)) {
        split_ws(line, toks);
        if (toks.empty()) continue;
        transactions_str.push_back(toks);
        for (const auto& s : toks) counts[s] += 1;
    }
    f.close();

    // Frequent items
    std::vector<std::pair<std::string, int64_t>> frequent;
    frequent.reserve(counts.size());
    for (const auto& kv : counts) if (kv.second >= min_count) frequent.push_back(kv);

    std::sort(frequent.begin(), frequent.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });

    int num_items = (int)frequent.size();
    std::vector<std::string> id_to_item(num_items);
    std::unordered_map<std::string, int> item_to_id;
    item_to_id.reserve((size_t)num_items * 2);

    for (int i = 0; i < num_items; i++) {
        id_to_item[i] = frequent[i].first;
        item_to_id[id_to_item[i]] = i; // id order = global order
    }

    std::vector<int> global_rank_order(num_items);
    for (int i = 0; i < num_items; i++) global_rank_order[i] = i;

    // ------------------------------------------------------------
    // Convert transactions to IDs (filter infrequent), sort, unique
    // ------------------------------------------------------------
    std::vector<std::vector<int>> transactions;
    transactions.reserve(transactions_str.size());

    for (const auto& tx : transactions_str) {
        std::vector<int> ids;
        ids.reserve(tx.size());
        for (const auto& s : tx) {
            auto it = item_to_id.find(s);
            if (it != item_to_id.end()) ids.push_back(it->second);
        }
        if (ids.empty()) continue;

        std::sort(ids.begin(), ids.end());           // already in frequency order via ID assignment
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        if (!ids.empty()) transactions.push_back(std::move(ids));
    }

    // ------------------------------------------------------------
    // Build projected DB with AVX2 bulk copies (one flat vector per base item)
    // ------------------------------------------------------------
    std::vector<std::vector<int>> projected_flat;
    build_projected_db_avx(transactions, num_items, projected_flat);

    // ------------------------------------------------------------
    // Parallel mine each base item using OpenMP
    // ------------------------------------------------------------
    int64_t total_patterns = 0;

#pragma omp parallel
    {
        int64_t local_patterns = 0;
        std::vector<std::pair<std::vector<int>, int64_t>> paths;
        std::vector<int> suffix;

#pragma omp for schedule(dynamic, 1)
        for (int base = 0; base < num_items; base++) {
            // singleton is always a pattern (frequent 1-item)
            local_patterns += 1;

            // unpack conditional pattern base (prefixes for base)
            if (projected_flat[base].empty()) continue;

            unpack_projected_db(projected_flat[base], paths);

            FPTree* ct = build_conditional_tree(paths, num_items, min_count, global_rank_order);
            if (!ct) continue;

            suffix.clear();
            suffix.push_back(base);

            fp_growth_mine(*ct, min_count, suffix, local_patterns, global_rank_order);

            delete ct;
        }

#pragma omp atomic
        total_patterns += local_patterns;
    }

    double t_end = omp_get_wtime();

    std::cout << "FP-Growth (OpenMP + AVX2) finished.\n";
    std::cout << "Transactions: " << total_lines << "\n";
    std::cout << "Min support: " << min_support << " => min_count = " << min_count << "\n";
    std::cout << "Frequent items: " << num_items << "\n";
    std::cout << "Total patterns counted (includes all frequent 1-itemsets): " << total_patterns << "\n";
    std::cout << "Total execution time: " << (t_end - t_start) << " seconds\n";
        auto end = std::chrono::high_resolution_clock::now();

    auto duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(end-start);

    std::cout<<"\nExecution Time: "
             <<duration.count()<<" ms\n";
    return 0;
}
