// 递归插入：返回插入后的根节点（处理空树场景）
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};
TreeNode* insertBST_Recur(TreeNode* root, int val) {
    // 终止条件：找到空位置，创建新节点并返回
    if (root == nullptr) {
        return new TreeNode(val);
    }

    // 递归查找插入位置
    if (val < root->val) {
        // 新值更小，插入左子树
        root->left = insertBST_Recur(root->left, val);
    } else if (val > root->val) {
        // 新值更大，插入右子树
        root->right = insertBST_Recur(root->right, val);
    }
    // 若值相等，直接返回原节点（不插入重复值）
    return root;
}