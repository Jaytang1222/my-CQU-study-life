#include<iostream>
#include<vector>
#include<queue>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* make_treenode(std::vector<int>& input, int start, int end) {
    if (start > end) return nullptr;
    int rootval = input[start];
    TreeNode* root = new TreeNode(rootval);
    int mid = start + 1;
    while (mid <= end && input[mid] < rootval) {
        mid++;
    }
    root->left = make_treenode(input, start + 1, mid - 1);
    root->right = make_treenode(input, mid, end);
    return root;
}

TreeNode* deleteBST(TreeNode* root, int deleteval) {
    if (!root) return nullptr;
    if (root->val > deleteval) {
        root->left = deleteBST(root->left, deleteval);
    } else if (root->val < deleteval) {
        root->right = deleteBST(root->right, deleteval);
    } else {
        if (!root->left && !root->right) {
            delete root;
            return nullptr;
        } else if (root->left) {
            TreeNode* deletenode = root->left;
            while (deletenode->right) {
                deletenode = deletenode->right;
            }
            root->val = deletenode->val;
            root->left = deleteBST(root->left, deletenode->val);
        } else if (!root->left && root->right) {
            TreeNode* deletenode = root->right;
            while (deletenode->left) {
                deletenode = deletenode->left;
            }
            root->val = deletenode->val;
            root->right = deleteBST(root->right, deletenode->val);
        }
    }
    return root;
}

void leveltraversal(TreeNode* root, std::vector<int>& path) {
    std::queue<TreeNode*> que;
    if (root) que.push(root);
    while (!que.empty()) {
        int size = que.size();
        for (int i = 0; i < size; i++) {
            TreeNode* node = que.front();
            que.pop();
            path.push_back(node->val);
            if (node->left) que.push(node->left);
            if (node->right) que.push(node->right);
        }
    }
    return;
}

void printleveltraversal(std::vector<int>& path) {
    int n = path.size();
    for (int i = 0; i < n - 1; i++) {
        std::cout << path[i] << " ";
    }
    std::cout << path[n- 1] << '\n';
}


int main() {
    int n;
    std::cin >> n;
    std::vector<int> input(n);
    for (int i = 0; i < n; i++) {
        std::cin >> input[i];
    }
    int m;
    std::cin >> m;
    std::vector<int> deletenode(m);
    for (int i = 0; i < m; i++) {
        std::cin >> deletenode[i];
    }
    TreeNode* root = make_treenode(input, 0, n - 1);
    std::vector<int> path;
    for (int num : deletenode) {
        root = deleteBST(root, num);
    }
    leveltraversal(root, path);
    printleveltraversal(path);
    return 0;
}