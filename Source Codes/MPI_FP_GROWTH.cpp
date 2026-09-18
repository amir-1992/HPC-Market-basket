#define _CRT_SECURE_NO_WARNINGS
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <chrono>

using namespace std;

/* ---------------- utilities ---------------- */

int count_lines(const char* file)
{
    ifstream f(file);
    string line;
    int c = 0;
    while (getline(f,line)) c++;
    return c;
}

void compute_range(int total,int rank,int size,int& start,int& end)
{
    int chunk = total / size;

    start = rank * chunk;

    if(rank == size-1)
        end = total;
    else
        end = start + chunk;
}

void read_transactions(const char* file,
                       int start,
                       int end,
                       vector<vector<string>>& transactions)
{
    ifstream f(file);
    string line;
    int index=0;

    while(getline(f,line))
    {
        if(index>=start && index<end)
        {
            vector<string> t;
            string item;
            stringstream ss(line);

            while(ss>>item)
                t.push_back(item);

            transactions.push_back(t);
        }

        if(index>=end) break;

        index++;
    }
}

/* ---------------- main ---------------- */

int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    auto start_time = chrono::high_resolution_clock::now();

    const char* file =
    "C:/Users/Every One/Desktop/parallel_paper/HPC-Apriori-main/order_products__prior_0.50.txt";

    double min_support = 0.001;

    /* -------- dataset partition -------- */

    int total_lines = count_lines(file);

    int start,end;
    compute_range(total_lines,rank,size,start,end);

    vector<vector<string>> transactions;

    read_transactions(file,start,end,transactions);

    /* -------- local frequency -------- */

    unordered_map<string,int> local_counts;

    for(auto& t:transactions)
        for(auto& item:t)
            local_counts[item]++;

    /* -------- dictionary construction -------- */

    vector<string> local_keys;

    for(auto& p:local_counts)
        local_keys.push_back(p.first);

    int local_n = local_keys.size();

    vector<int> sizes(size);

    MPI_Gather(&local_n,1,MPI_INT,
               sizes.data(),1,MPI_INT,
               0,MPI_COMM_WORLD);

    vector<string> global_keys;

    if(rank==0)
    {
        global_keys.insert(global_keys.end(),
                           local_keys.begin(),
                           local_keys.end());

        for(int src=1; src<size; src++)
        {
            int n = sizes[src];

            for(int j=0;j<n;j++)
            {
                int len;

                MPI_Recv(&len,1,MPI_INT,src,0,
                         MPI_COMM_WORLD,MPI_STATUS_IGNORE);

                string s(len,' ');

                MPI_Recv(&s[0],len,MPI_CHAR,src,0,
                         MPI_COMM_WORLD,MPI_STATUS_IGNORE);

                global_keys.push_back(s);
            }
        }

        sort(global_keys.begin(),global_keys.end());
        global_keys.erase(unique(global_keys.begin(),
                                 global_keys.end()),
                                 global_keys.end());
    }
    else
    {
        for(auto& s:local_keys)
        {
            int len = s.size();

            MPI_Send(&len,1,MPI_INT,0,0,MPI_COMM_WORLD);
            MPI_Send(s.c_str(),len,MPI_CHAR,0,0,MPI_COMM_WORLD);
        }
    }

    /* -------- broadcast dictionary -------- */

    int dict_size;

    if(rank==0)
        dict_size = global_keys.size();

    MPI_Bcast(&dict_size,1,MPI_INT,0,MPI_COMM_WORLD);

    if(rank!=0)
        global_keys.resize(dict_size);

    for(int i=0;i<dict_size;i++)
    {
        int len;

        if(rank==0)
            len = global_keys[i].size();

        MPI_Bcast(&len,1,MPI_INT,0,MPI_COMM_WORLD);

        if(rank!=0)
            global_keys[i].resize(len);

        MPI_Bcast(&global_keys[i][0],
                  len,
                  MPI_CHAR,
                  0,
                  MPI_COMM_WORLD);
    }

    /* -------- indexing -------- */

    unordered_map<string,int> index;

    for(int i=0;i<dict_size;i++)
        index[global_keys[i]] = i;

    vector<int> local_array(dict_size,0);

    for(auto& p:local_counts)
        local_array[index[p.first]] = p.second;

    vector<int> global_array(dict_size,0);

    MPI_Allreduce(local_array.data(),
                  global_array.data(),
                  dict_size,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    /* -------- support filtering -------- */

    int min_count = min_support * total_lines;

    int frequent_items = 0;

    for(int i=0;i<dict_size;i++)
        if(global_array[i] >= min_count)
            frequent_items++;

    auto end_time = chrono::high_resolution_clock::now();

    auto duration =
    chrono::duration_cast<chrono::milliseconds>(end_time-start_time);

    if(rank==0)
    {
        cout<<"Frequent items: "<<frequent_items<<endl;
        cout<<"Execution time: "<<duration.count()<<" ms"<<endl;
    }

    MPI_Finalize();

    return 0;
}
