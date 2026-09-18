#define _CRT_SECURE_NO_WARNINGS

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>

using namespace std;

/* ============================================================
   Portable 64-bit popcount (no compiler intrinsics needed)
============================================================ */

static inline int popcount64(uint64_t x)
{
    int count = 0;
    while (x)
    {
        x &= x - 1;
        count++;
    }
    return count;
}

/* ============================================================
   Split tokens
============================================================ */

static void split_ws(const string& line, vector<string>& out)
{
    out.clear();

    istringstream iss(line);
    string tok;

    while (iss >> tok)
        out.push_back(tok);
}

/* ============================================================
   AVX popcount for 256-bit block
============================================================ */

static inline int popcount256(__m256i v)
{
    uint64_t tmp[4];

    _mm256_storeu_si256((__m256i*)tmp, v);

    return
        popcount64(tmp[0]) +
        popcount64(tmp[1]) +
        popcount64(tmp[2]) +
        popcount64(tmp[3]);
}

/* ============================================================
   AVX support counting
============================================================ */

static int avx_support(
    const vector<uint64_t>& A,
    const vector<uint64_t>& B)
{
    int support = 0;

    int blocks = (int)A.size();
    int i = 0;

    for (; i + 4 <= blocks; i += 4)
    {
        __m256i a = _mm256_loadu_si256((__m256i*)&A[i]);
        __m256i b = _mm256_loadu_si256((__m256i*)&B[i]);

        __m256i r = _mm256_and_si256(a, b);

        support += popcount256(r);
    }

    for (; i < blocks; i++)
        support += popcount64(A[i] & B[i]);

    return support;
}

/* ============================================================
   Main
============================================================ */

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        cout << "Usage: Only_AVX.exe dataset min_support\n";
        return 1;
    }

    auto start = chrono::high_resolution_clock::now();

    const char* file = argv[1];
    double min_support = atof(argv[2]);

    ifstream f(file);

    if (!f)
    {
        cout << "File open error\n";
        return 1;
    }

    unordered_map<string, int> item_id;
    vector<vector<int>> transactions;

    string line;
    vector<string> toks;

    /* ------------------------------------------------
       Read dataset
    ------------------------------------------------ */

    while (getline(f, line))
    {
        split_ws(line, toks);

        if (toks.empty())
            continue;

        vector<int> t;

        for (auto& s : toks)
        {
            if (!item_id.count(s))
                item_id[s] = (int)item_id.size();

            t.push_back(item_id[s]);
        }

        transactions.push_back(t);
    }

    int T = (int)transactions.size();
    int I = (int)item_id.size();

    int min_count = (int)ceil(min_support * T);

    int blocks = (T + 63) / 64;

    /* ------------------------------------------------
       Vertical bitsets
    ------------------------------------------------ */

    vector<vector<uint64_t>> bitsets(
        I,
        vector<uint64_t>(blocks, 0)
    );

    for (int tid = 0; tid < T; tid++)
    {
        for (int item : transactions[tid])
        {
            bitsets[item][tid / 64] |=
                1ULL << (tid % 64);
        }
    }

    /* ------------------------------------------------
       L1 frequent items
    ------------------------------------------------ */

    vector<int> L1;

    for (int i = 0; i < I; i++)
    {
        int sup = 0;

        for (auto b : bitsets[i])
            sup += popcount64(b);

        if (sup >= min_count)
            L1.push_back(i);
    }

    long long patterns = (long long)L1.size();

    /* ------------------------------------------------
       L2 using AVX
    ------------------------------------------------ */

    for (size_t i = 0; i < L1.size(); i++)
    {
        for (size_t j = i + 1; j < L1.size(); j++)
        {
            int sup = avx_support(
                bitsets[L1[i]],
                bitsets[L1[j]]
            );

            if (sup >= min_count)
                patterns++;
        }
    }

    auto end = chrono::high_resolution_clock::now();

    auto ms =
        chrono::duration_cast<chrono::milliseconds>(
            end - start
        );

    cout << "Transactions : " << T << "\n";
    cout << "Items        : " << I << "\n";
    cout << "Min Support  : " << min_count << "\n";
    cout << "Patterns     : " << patterns << "\n";
    cout << "Time (ms)    : " << ms.count() << "\n";

    return 0;
}
