#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <chrono>

using namespace std;

const float MIN_SUPPORT = 0.001f;

struct Transaction
{
    vector<int> items;
};

void read_dataset(const string& filename,
                  vector<vector<string>>& raw_transactions);

void build_dictionary(const vector<vector<string>>& raw,
                      unordered_map<string,int>& item_to_id,
                      vector<string>& id_to_item,
                      vector<int>& freq);

void convert_transactions(const vector<vector<string>>& raw,
                          const unordered_map<string,int>& item_to_id,
                          vector<Transaction>& transactions);

void count_support_parallel(const vector<Transaction>& transactions,
                            vector<int>& global_count);

int main(int argc,char* argv[])
{
    if(argc < 3)
    {
        cout << "Usage: program <threads> <dataset_path>\n";
        return 0;
    }

    int threads = atoi(argv[1]);
    string dataset = argv[2];

    omp_set_num_threads(threads);

    cout << "Threads: " << threads << endl;
    cout << "Dataset: " << dataset << endl;

    auto start = chrono::high_resolution_clock::now();

    vector<vector<string>> raw_transactions;

    read_dataset(dataset, raw_transactions);

    cout << "Transactions: " << raw_transactions.size() << endl;

    unordered_map<string,int> item_to_id;
    vector<string> id_to_item;
    vector<int> freq;

    build_dictionary(raw_transactions,
                     item_to_id,
                     id_to_item,
                     freq);

    vector<Transaction> transactions;

    convert_transactions(raw_transactions,
                         item_to_id,
                         transactions);

    vector<int> global_count(id_to_item.size(),0);

    count_support_parallel(transactions,global_count);

    int transaction_count = transactions.size();

    vector<string> frequent_items;

#pragma omp parallel
    {
        vector<string> local;

#pragma omp for nowait
        for(int i=0;i<global_count.size();i++)
        {
            float support =
            (float)global_count[i]/transaction_count;

            if(support >= MIN_SUPPORT)
                local.push_back(id_to_item[i]);
        }

#pragma omp critical
        frequent_items.insert(frequent_items.end(),
                              local.begin(),
                              local.end());
    }

    auto end = chrono::high_resolution_clock::now();

    auto dur =
    chrono::duration_cast<chrono::milliseconds>(end-start);

    cout << "Frequent items: " << frequent_items.size() << endl;
    cout << "Execution time(ms): " << dur.count() << endl;

    return 0;
}


void read_dataset(const string& filename,
                  vector<vector<string>>& raw_transactions)
{
    ifstream file(filename);

    if(!file.is_open())
    {
        cout << "Cannot open dataset\n";
        exit(0);
    }

    string line;

    while(getline(file,line))
    {
        if(line.empty()) continue;

        vector<string> row;

        stringstream ss(line);
        string item;

        while(getline(ss,item,' '))
        {
            if(!item.empty())
                row.push_back(item);
        }

        if(!row.empty())
            raw_transactions.push_back(row);
    }

    file.close();
}


void build_dictionary(const vector<vector<string>>& raw,
                      unordered_map<string,int>& item_to_id,
                      vector<string>& id_to_item,
                      vector<int>& freq)
{
    unordered_map<string,int> count;

    for(const auto& row : raw)
        for(const auto& item : row)
            count[item]++;

    int id = 0;

    for(auto &p : count)
    {
        item_to_id[p.first] = id++;
        id_to_item.push_back(p.first);
        freq.push_back(p.second);
    }
}


void convert_transactions(const vector<vector<string>>& raw,
                          const unordered_map<string,int>& item_to_id,
                          vector<Transaction>& transactions)
{
    transactions.resize(raw.size());

#pragma omp parallel for
    for(int i=0;i<raw.size();i++)
    {
        for(const auto& item : raw[i])
        {
            int id = item_to_id.at(item);
            transactions[i].items.push_back(id);
        }

        sort(transactions[i].items.begin(),
             transactions[i].items.end());
    }
}


void count_support_parallel(const vector<Transaction>& transactions,
                            vector<int>& global_count)
{
    int item_count = global_count.size();

    vector<vector<int>> local_counts(
        omp_get_max_threads(),
        vector<int>(item_count,0));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();

#pragma omp for schedule(static)
        for(int t=0;t<transactions.size();t++)
        {
            for(int item : transactions[t].items)
                local_counts[tid][item]++;
        }
    }

    for(auto &lc : local_counts)
        for(int i=0;i<item_count;i++)
            global_count[i] += lc[i];
}
