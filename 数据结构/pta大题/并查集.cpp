#include<iostream>
#include<vector>

using namespace std;

int k = 1e5 + 1;

vector<int> father(k);

void init() {
    for (int i = 0; i < k; i++) {
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

int main() {
    int n, m;
    cin >> n >> m;
    init();
    for (int i = 0;i < m; i++) {
        int z, x, y;
        cin >> z >> x >> y;
        if (z == 1) {
            join(x, y);
        } else if (z == 2) {
            if (isinsame(x, y)) {
                cout << "Y" << endl;
            } else {
                cout << "N" << endl;
            }
        }
    }
    return 0;
}