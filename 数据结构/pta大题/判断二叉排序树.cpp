#include<iostream>
#include<vector>

using namespace std;

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* makeroot(vector<char>& input, int& start) {
    if (start >= input.size()) return nullptr;
    char cur = input[start];
    start++;
    if (cur == '*') {
        return nullptr;
    } 
    int temp = cur - '0';
    TreeNode* root = new TreeNode(temp);
    root->left = makeroot(input, start);
    root->right = makeroot(input, start);
    return root;
}

void midorder(TreeNode* root, vector<int>& path) {
    if (!root) return;
    midorder(root->left, path);
    path.push_back(root->val);
    midorder(root->right, path);
}

bool isok(vector<int>& path) {
    for (int i = 1; i < path.size(); i++) {
        if (path[i] < path[i - 1]) return false;
    }
    return true;
}

int main() {
    string s;
    while (cin >> s) {
        vector<char> input(s.begin(), s.end());
        vector<int> path;
        int start = 0;
        TreeNode* root = makeroot(input, start);
        midorder(root, path);
        if (isok(path)) {
            cout << "YES" << endl;
        } else {
            cout << "NO" << endl;
        }
    }
    return 0;
}

