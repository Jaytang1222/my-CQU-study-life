void traversal(TreeNode* root) {
    if (!root) return;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        TreeNode* cur = q.front();
        q.pop();
        cout << cur->val;
        if (cur->left) q.push(cur->left);
        if (cur->right) q.push(cur->right);
    }
}

