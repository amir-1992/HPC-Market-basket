// fp_growth_cuda_heavy_no_lambda.cu
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>

// ==========================
// CUDA error check helper
// ==========================
#define CUDA_CHECK(call) do {                                  \
  cudaError_t e = (call);                                      \
  if (e != cudaSuccess) {                                      \
    std::cerr << "CUDA error: " << cudaGetErrorString(e)       \
              << " at " << __FILE__ << ":" << __LINE__ << "\n";\
    std::exit(1);                                              \
  }                                                            \
} while (0)

// ============================================================
// Kernel: count 1-item supports (transactions are flattened CSR)
// ============================================================
__global__ void count_support_kernel(
    const int* flat_items,
    const int* offsets,
    int num_transactions,
    int* item_counts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_transactions) return;

    int start = offsets[tid];
    int end   = offsets[tid + 1];
    for (int i = start; i < end; i++) {
        int item = flat_items[i];
        atomicAdd(&item_counts[item], 1);
    }
}

// ============================================================
// Kernel: map item -> rank (or -1 if infrequent) for each entry
// ============================================================
__global__ void map_to_rank_kernel(
    const int* in_items,
    int* out_ranks,
    const int* item_to_rank,
    int n_entries)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entries) return;
    int item = in_items[i];
    out_ranks[i] = item_to_rank[item]; // -1 if infrequent
}

// ============================================================
// Thrust functors (NO extended lambdas needed)
// ============================================================

struct rank_is_negative {
    __host__ __device__
    bool operator()(const thrust::tuple<int,int>& x) const {
        return thrust::get<1>(x) < 0;
    }
};

struct make_key_functor {
    __host__ __device__
    unsigned long long operator()(const thrust::tuple<int,int>& x) const {
        unsigned long long tx = (unsigned long long)(unsigned int)thrust::get<0>(x);
        unsigned long long rk = (unsigned long long)(unsigned int)thrust::get<1>(x);
        return (tx << 32) | rk;
    }
};

struct extract_txid_functor {
    __host__ __device__
    int operator()(unsigned long long k) const {
        return (int)(k >> 32);
    }
};

// ============================================================
// FP-tree (CPU) + very simple mining
// (prints patterns along root-to-node paths)
// ============================================================
struct FPNode {
    int item;
    int count;
    FPNode* parent;
    std::unordered_map<int, FPNode*> children;
    FPNode(int it, FPNode* p) : item(it), count(1), parent(p) {}
};

static void insert_fp(FPNode* root, const std::vector<int>& tx) {
    FPNode* cur = root;
    for (int item : tx) {
        auto it = cur->children.find(item);
        if (it != cur->children.end()) {
            it->second->count++;
            cur = it->second;
        } else {
            FPNode* nn = new FPNode(item, cur);
            cur->children[item] = nn;
            cur = nn;
        }
    }
}

static FPNode* build_fp_tree_cpu(const std::vector<std::vector<int>>& transactions_ranked) {
    FPNode* root = new FPNode(-1, nullptr);
    root->count = 0;
    for (const auto& tx : transactions_ranked) insert_fp(root, tx);
    return root;
}

static void mine_fp_simple(FPNode* node, std::vector<int>& prefix) {
    for (auto& kv : node->children) {
        FPNode* child = kv.second;
        prefix.push_back(child->item);

        std::cout << "Pattern: ";
        for (int x : prefix) std::cout << x << ' ';
        std::cout << "(count=" << child->count << ")\n";

        mine_fp_simple(child, prefix);
        prefix.pop_back();
    }
}

// ============================================================
// Read transactions (integers per line)
// ============================================================
static void read_transactions(
    const std::string& filename,
    std::vector<std::vector<int>>& transactions,
    int& max_item)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file\n";
        std::exit(1);
    }

    std::string line;
    max_item = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int item;
        std::vector<int> tx;

        while (iss >> item) {
            tx.push_back(item);
            max_item = std::max(max_item, item);
        }

        if (!tx.empty())
            transactions.push_back(std::move(tx));
    }
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv) {
    using Clock = std::chrono::high_resolution_clock;
    auto t_total0 = Clock::now();

    if (argc < 3) {
        std::cout << "Usage: fp_growth_cuda_heavy.exe input.txt min_support\n";
        return 1;
    }

    std::string filename = argv[1];
    float min_support = std::stof(argv[2]);

    // ----------------------------
    // Read on CPU
    // ----------------------------
    std::vector<std::vector<int>> transactions;
    int max_item = 0;
    read_transactions(filename, transactions, max_item);

    int num_transactions = (int)transactions.size();
    int num_items = max_item + 1;
    int min_count = (int)std::ceil(min_support * num_transactions);

    std::cout << "Transactions: " << num_transactions << "\n";
    std::cout << "Items:        " << num_items << "\n";
    std::cout << "Min count:    " << min_count << "\n";

    // ----------------------------
    // Flatten (CSR) on CPU
    // ----------------------------
    std::vector<int> h_offsets(num_transactions + 1);
    std::vector<int> h_flat;
    h_flat.reserve(1024);

    int idx = 0;
    for (int t = 0; t < num_transactions; t++) {
        h_offsets[t] = idx;
        for (int item : transactions[t]) {
            h_flat.push_back(item);
            idx++;
        }
    }
    h_offsets[num_transactions] = idx;

    int n_entries = (int)h_flat.size();

    // ============================
    // GPU preprocessing timing
    // ============================
    cudaEvent_t g0, g1;
    CUDA_CHECK(cudaEventCreate(&g0));
    CUDA_CHECK(cudaEventCreate(&g1));
    CUDA_CHECK(cudaEventRecord(g0));

    // ----------------------------
    // Copy to GPU
    // ----------------------------
    thrust::device_vector<int> d_flat(h_flat.begin(), h_flat.end());
    thrust::device_vector<int> d_offsets(h_offsets.begin(), h_offsets.end());

    // ----------------------------
    // 1) Count 1-item supports (GPU)
    // ----------------------------
    thrust::device_vector<int> d_counts(num_items, 0);

    int block = 256;
    int grid = (num_transactions + block - 1) / block;

    count_support_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_flat.data()),
        thrust::raw_pointer_cast(d_offsets.data()),
        num_transactions,
        thrust::raw_pointer_cast(d_counts.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy counts back (small-ish) to build rank table
    thrust::host_vector<int> h_counts = d_counts;

    // ----------------------------
    // Build item->rank (descending frequency) on CPU
    // Only frequent items get a rank [0..F-1], others -1
    // ----------------------------
    std::vector<int> frequent_items;
    frequent_items.reserve(num_items);

    for (int i = 0; i < num_items; i++) {
        if (h_counts[i] >= min_count)
            frequent_items.push_back(i);
    }

    std::sort(frequent_items.begin(), frequent_items.end(),
              [&](int a, int b) { return h_counts[a] > h_counts[b]; });

    std::vector<int> h_item_to_rank(num_items, -1);
    for (int r = 0; r < (int)frequent_items.size(); r++) {
        h_item_to_rank[frequent_items[r]] = r;
    }

    int F = (int)frequent_items.size();
    std::cout << "Frequent 1-items: " << F << "\n";

    // ----------------------------
    // 2) Map each entry item->rank on GPU (infrequent => -1)
    // ----------------------------
    thrust::device_vector<int> d_item_to_rank(h_item_to_rank.begin(), h_item_to_rank.end());
    thrust::device_vector<int> d_ranks(n_entries);

    int grid2 = (n_entries + block - 1) / block;
    map_to_rank_kernel<<<grid2, block>>>(
        thrust::raw_pointer_cast(d_flat.data()),
        thrust::raw_pointer_cast(d_ranks.data()),
        thrust::raw_pointer_cast(d_item_to_rank.data()),
        n_entries);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ----------------------------
    // 3) Build (tx_id, rank) pairs and remove infrequent (-1) on GPU
    // ----------------------------
    thrust::device_vector<int> d_txid(n_entries);

    // Expand txid on CPU once, then copy (simple & reliable)
    std::vector<int> h_txid(n_entries);
    for (int t = 0; t < num_transactions; t++) {
        for (int i = h_offsets[t]; i < h_offsets[t + 1]; i++) {
            h_txid[i] = t;
        }
    }
    d_txid.assign(h_txid.begin(), h_txid.end());

    // Remove where rank == -1
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_txid.begin(), d_ranks.begin()));
    auto zip_end   = thrust::make_zip_iterator(thrust::make_tuple(d_txid.end(),   d_ranks.end()));

    auto new_end = thrust::remove_if(zip_begin, zip_end, rank_is_negative());

    int new_n_entries = (int)(new_end - zip_begin);
    d_txid.resize(new_n_entries);
    d_ranks.resize(new_n_entries);

    // ----------------------------
    // 4) Sort by (txid, rank) on GPU -> within each tx, ranks sorted ascending
    // (ascending rank == descending support)
    // ----------------------------
    thrust::device_vector<unsigned long long> d_key(new_n_entries);

    auto zip2_begin = thrust::make_zip_iterator(thrust::make_tuple(d_txid.begin(), d_ranks.begin()));
    auto zip2_end   = thrust::make_zip_iterator(thrust::make_tuple(d_txid.end(),   d_ranks.end()));

    thrust::transform(zip2_begin, zip2_end, d_key.begin(), make_key_functor());

    thrust::sort_by_key(d_key.begin(), d_key.end(), d_ranks.begin());

    // ----------------------------
    // 5) Rebuild compacted CSR offsets on GPU (counts per tx)
    // ----------------------------
    thrust::device_vector<int> d_txid_sorted(new_n_entries);
    thrust::transform(d_key.begin(), d_key.end(), d_txid_sorted.begin(), extract_txid_functor());

    thrust::device_vector<int> d_unique_tx(num_transactions);
    thrust::device_vector<int> d_len(num_transactions);

    auto ones = thrust::make_constant_iterator(1);
    auto red_end = thrust::reduce_by_key(
        d_txid_sorted.begin(), d_txid_sorted.end(),
        ones,
        d_unique_tx.begin(),
        d_len.begin());

    int n_unique = (int)(red_end.first - d_unique_tx.begin());
    d_unique_tx.resize(n_unique);
    d_len.resize(n_unique);

    thrust::device_vector<int> d_new_offsets(num_transactions + 1, 0);

    thrust::scatter(d_len.begin(), d_len.end(),
                    d_unique_tx.begin(),
                    d_new_offsets.begin());

    thrust::exclusive_scan(d_new_offsets.begin(), d_new_offsets.end(),
                           d_new_offsets.begin());

    // Stop GPU timer
    CUDA_CHECK(cudaEventRecord(g1));
    CUDA_CHECK(cudaEventSynchronize(g1));
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, g0, g1));
    CUDA_CHECK(cudaEventDestroy(g0));
    CUDA_CHECK(cudaEventDestroy(g1));

    // ----------------------------
    // Copy compacted ranked DB back to CPU
    // ----------------------------
    thrust::host_vector<int> h_rank_items = d_ranks;
    thrust::host_vector<int> h_new_offsets = d_new_offsets;

    std::vector<std::vector<int>> ranked_transactions(num_transactions);

    for (int t = 0; t < num_transactions; t++) {
        int s = h_new_offsets[t];
        int e = h_new_offsets[t + 1];
        ranked_transactions[t].reserve(e - s);
        for (int i = s; i < e; i++) {
            ranked_transactions[t].push_back(h_rank_items[i]);
        }
    }

    // ----------------------------
    // Build FP-tree + mine (CPU)
    // ----------------------------
    auto t_tree0 = Clock::now();
    FPNode* root = build_fp_tree_cpu(ranked_transactions);
    std::vector<int> prefix;
    mine_fp_simple(root, prefix);
    auto t_tree1 = Clock::now();

    auto t_total1 = Clock::now();

    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_total1 - t_total0).count();
    auto tree_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t_tree1 - t_tree0).count();

    std::cout << "\n=== Timing ===\n";
    std::cout << "GPU preprocess (count+filter+sort+rebuild): " << gpu_ms << " ms\n";
    std::cout << "CPU FP-tree build + simple mining:         " << tree_ms << " ms\n";
    std::cout << "Total wall time:                           " << total_ms << " ms\n";

    return 0;
}
