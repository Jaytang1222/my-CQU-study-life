#include<iostream>
#include<vector>
#include<unordered_map>

using namespace std;

unordered_map<char, int>mymap;

struct TreeNode{
    char val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(char x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* make_root(vector<char>& prevec, vector<char>& midvec, int prestart, int preend, int midstart, int midend) {
    if (prestart > preend || midstart > midend) {
        return nullptr;
    }
    TreeNode* root = new TreeNode(prevec[prestart]);
    int splitindex = mymap[root->val];
    int leftsize = splitindex - midstart;
    root->left = make_root(prevec, midvec, prestart + 1, prestart + leftsize, midstart, splitindex - 1);
    root->right = make_root(prevec, midvec, prestart + leftsize + 1, preend, splitindex + 1, midend);
    return root;
}

void postorder(TreeNode* root) {
    if (!root) return;
    postorder(root->left);
    postorder(root->right);
    cout << root->val;
}


int main() {
    string a, b;
    cin >> a >> b;
    int n = a.size();
    vector<char>prevec(a.begin(), a.end());
    vector<char>midvec(b.begin(), b.end());
    for (int i = 0; i < n; i++) {
        auto it = make_pair(b[i], i);
        mymap.insert(it);
    }
    TreeNode* root = make_root(prevec, midvec, 0, n - 1, 0, n - 1);
    postorder(root);
    return 0;
}