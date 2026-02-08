#include<iostream>
#include<vector>
#include<unordered_map>
#include<stack>

using namespace std;

void printvec(vector<int>& res) {
    for (int num : res) {
        cout << num << " ";
    }
    cout << endl;
}

void tuopusort(unordered_map<int, vector<int>>& mymap, vector<int>& indegree, vector<int>& res) {
    stack<int> st;
    for (int i = 1; i < indegree.size(); i++) {
        if (indegree[i] == 0) st.push(i);
    }
    while (st.size()) {
        int temp = st.top();
        st.pop();
        //visitied[temp] = true;
        res.push_back(temp);
        vector<int> files = mymap[temp];
        for (int i = 0; i < int(files.size()); i++) {
            indegree[files[i]]--;
            if (indegree[files[i]] == 0) {
                st.push(files[i]);
                //visitied[indegree[files[i]]] = true;
            }
        }
    }
}

int main() {
    int n, m;
    cin >> n >> m;
    unordered_map<int, vector<int>> mymap;
    //vector<bool> visitied(n + 1, false);
    vector<int> indegree(n + 1, 0);
    while (m--) {
        int s, t;
        cin >> s >> t;
        mymap[s].insert(mymap[s].begin(), t);
        indegree[t]++;
    }
    vector<int> res;
    tuopusort(mymap, indegree, res);
    if (int(res.size()) == n) {
        printvec(res);
    } else {
        printvec(res);
        cout << 0 << endl;
    }
    return 0;
    
}