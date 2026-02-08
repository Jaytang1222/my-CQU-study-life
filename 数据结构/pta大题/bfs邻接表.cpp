#include<iostream>
#include<vector>
#include<queue>
#include<algorithm>

using namespace std;

void printvec(vector<int>& res) {
    for (int num : res) {
        cout << num << " ";
    }
    cout << endl;
}

void bfs(int start, vector<vector<int>>& adj, vector<int>& res, vector<bool>& visitied, int& resnum, int n) {
    for (int i = 0; i < n; i++) {
        if (!visitied[i]) {
            resnum++;
            visitied[i] = true;
            res.push_back(i);
            queue<int> q;
            q.push(i);
            while (q.size()) {
                int temp = q.front();
                q.pop();
                for (int num : adj[temp]) {
                    if (!visitied[num]) {
                        visitied[num] = true;
                        q.push(num);
                        res.push_back(num);
                    }
                }
            }
        }
    }
}

int main() {
    int n, e;
    cin >> n >> e;
    int rese = e;
    vector<vector<int>> adj(n);
    while (e--) {
        int t1, t2;
        cin >> t1 >> t2;
        adj[t1].push_back(t2);
        adj[t2].push_back(t1);
    }
    for (int i = 0; i < n; i++) {
        sort(adj[i].begin(), adj[i].end());
    }
    vector<int> res;
    vector<bool> visitied(n, false);
    int resnum = 0;
    bfs(0, adj, res, visitied, resnum, n);
    printvec(res);
    cout << resnum << endl;
    cout << rese << endl;
}