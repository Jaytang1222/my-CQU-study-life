#include<bits/stdc++.h>
#define ll long long

using namespace std;

int maxsize = 2e5 + 1;

vector<int>father (maxsize);

void init() {
    for (int i = 0; i < maxsize; i++) {
        father[i] = i;
    }
}

int find(int u) {
    return u == father[u] ? u : father[u] = find(father[u]);
}

void join(int u, int v) {
    u = find(u);
    v = find(v);
    if (u == v) return;
    father[v] = u;
}

bool isinsame(int u, int v) {
    u = find(u);
    v = find(v);
    return u == v;
}

struct edge{
    int l;
    int r;
    int val;
    edge(int u, int v, int w) : l(u), r(v), val(w) {}
};

int main() {
    int n, m;
    cin >> n >> m;
    ll res = 0;
    vector<edge> edges;
    while(m--) {
        int u, v, w;
        cin >> u >> v >> w;
        edges.push_back({u, v, w});
    }
    sort(edges.begin(), edges.end(), [](const edge& a, const edge& b) {
        return a.val < b.val;
    });
    init();
    for (edge e : edges) {
        if (isinsame(e.l, e.r)) continue;
        join(e.l, e.r);
        res += e.val;
    }
    cout << res << endl;
}