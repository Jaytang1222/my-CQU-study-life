#include<iostream>
#include<vector>

using namespace std;

struct Node{
    char val;
    Node* left;
    Node* right;
    Node(char x) : val(x), left(nullptr), right(nullptr) {}
};

Node* make_node(vector<char>& input, int& start) {
    if (start > input.size()) return nullptr;
    char cur = input[start];
    start++;
    if (cur == '#') {
        return nullptr;
    }
    Node* root = new Node(cur);
    root->left = make_node(input, start);
    root->right = make_node(input, start);
    return root;
}

void preorder(Node* root) {
    if (!root) return;
    cout << root->val;
    preorder(root->left);
    preorder(root->right);
    //cout << endl;
}


void midorder(Node* root) {
    if (!root) return;
    midorder(root->left);
    cout<< root->val;
    midorder(root->right);
    //cout << endl;
}


void postorder(Node* root) {
    if (!root) return;
    postorder(root->left);
    postorder(root->right);
    cout << root->val;
    //cout << endl;
}

int leaves(Node* root) {
    if (!root) return 0;
    int leftnum = leaves(root->left);
    int rightnum = leaves(root->right);
    if (!root->left && !root->right) {
        return leftnum + rightnum + 1;
    }
    return leftnum + rightnum;
}


int main() {
    string s;
    cin >> s;
    vector<char> input(s.begin(), s.end());
    int start = 0;
    Node* root = make_node(input, start);
    preorder(root);
    cout<<endl;
    midorder(root);
    cout << endl;
    postorder(root);
    cout << endl;
    cout << leaves(root) << endl;
    return 0;
}