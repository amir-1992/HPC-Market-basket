#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <sstream>
#include <chrono>

using namespace std;

struct FPNode
{
    string item;
    int count;
    FPNode* parent;
    unordered_map<string, FPNode*> children;
    FPNode* next;

    FPNode(string item, FPNode* parent)
    {
        this->item = item;
        this->count = 1;
        this->parent = parent;
        this->next = nullptr;
    }
};

struct FPTree
{
    FPNode* root;
    unordered_map<string, FPNode*> header_table;

    FPTree()
    {
        root = new FPNode("", nullptr);
        root->count = 0;
    }
};

void read_dataset(string filename, vector<vector<string>>& transactions)
{
    ifstream file(filename);
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
            transactions.push_back(row);
    }

    file.close();
}

void build_frequency(const vector<vector<string>>& transactions,
                     unordered_map<string,int>& freq)
{
    for(const auto& t : transactions)
        for(const auto& item : t)
            freq[item]++;
}

void insert_tree(vector<string>& items,
                 FPNode* node,
                 unordered_map<string,FPNode*>& header)
{
    if(items.empty()) return;

    string first = items[0];

    if(node->children.count(first))
    {
        node->children[first]->count++;
    }
    else
    {
        FPNode* newNode = new FPNode(first,node);
        node->children[first] = newNode;

        if(header.count(first)==0)
            header[first] = newNode;
        else
        {
            FPNode* temp = header[first];
            while(temp->next) temp = temp->next;
            temp->next = newNode;
        }
    }

    items.erase(items.begin());
    insert_tree(items,node->children[first],header);
}

FPTree build_tree(vector<vector<string>>& transactions,
                  unordered_map<string,int>& freq,
                  int min_support)
{
    FPTree tree;

    for(auto& t : transactions)
    {
        vector<string> filtered;

        for(auto& item : t)
            if(freq[item] >= min_support)
                filtered.push_back(item);

        sort(filtered.begin(),filtered.end(),
             [&](string a,string b)
             {
                 return freq[a] > freq[b];
             });

        insert_tree(filtered,tree.root,tree.header_table);
    }

    return tree;
}

vector<vector<string>> build_conditional_base(string item,
                                              unordered_map<string,FPNode*>& header)
{
    vector<vector<string>> base;

    FPNode* node = header[item];

    while(node)
    {
        int count = node->count;

        vector<string> path;
        FPNode* parent = node->parent;

        while(parent && parent->item!="")
        {
            path.push_back(parent->item);
            parent = parent->parent;
        }

        for(int i=0;i<count;i++)
            if(!path.empty())
                base.push_back(path);

        node = node->next;
    }

    return base;
}

void fp_growth(FPTree& tree,
               vector<string> prefix,
               unordered_map<string,int>& freq,
               int min_support)
{
    vector<pair<string,int>> items;

    for(auto& p : freq)
        if(p.second >= min_support)
            items.push_back(p);

    sort(items.begin(),items.end(),
         [](auto&a,auto&b){return a.second<b.second;});

    for(auto& p : items)
    {
        string item = p.first;

        vector<string> new_prefix = prefix;
        new_prefix.push_back(item);

        cout<<"Pattern: ";
        for(auto& s:new_prefix)
            cout<<s<<" ";
        cout<<" support="<<p.second<<endl;

        vector<vector<string>> cond_base =
            build_conditional_base(item,tree.header_table);

        unordered_map<string,int> cond_freq;

        build_frequency(cond_base,cond_freq);

        FPTree cond_tree =
            build_tree(cond_base,cond_freq,min_support);

        if(!cond_tree.header_table.empty())
            fp_growth(cond_tree,new_prefix,cond_freq,min_support);
    }
}

int main(int argc,char* argv[])
{
    if(argc<3)
    {
        cout<<"Usage: program dataset min_support_ratio\n";
        return 0;
    }

    string dataset = argv[1];
    float min_support_ratio = stof(argv[2]);

    auto start = chrono::high_resolution_clock::now();

    vector<vector<string>> transactions;

    read_dataset(dataset,transactions);

    int transaction_count = transactions.size();

    int min_support =
        (int)(transaction_count * min_support_ratio);

    unordered_map<string,int> freq;

    build_frequency(transactions,freq);

    FPTree tree = build_tree(transactions,freq,min_support);

    vector<string> prefix;

    fp_growth(tree,prefix,freq,min_support);

    auto end = chrono::high_resolution_clock::now();

    auto dur =
    chrono::duration_cast<chrono::milliseconds>(end-start);

    cout<<"Execution time(ms): "<<dur.count()<<endl;

    return 0;
}
