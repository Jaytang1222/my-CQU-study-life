#include<iostream>
#include<vector>
#include<stack>

using namespace std;

int op(char c) {
    if (c == '*' || c == '/') {
        return 2;
    } else if (c == '+' || c == '-') {
        return 1;
    }
    return 0;
}

void printstr(string& res) {
    int len = res.size();
    for (int i = 0; i < len - 1; i++) {
        cout << res[i] << " ";
    }
    cout << res[len - 1] << endl;
}

int main() {
    string s;
    cin >> s;
    string res;
    stack<char> temp;
    for (char c : s) {
        if (isdigit(c)) {
            res += c;
        } else if (c == '(') {
            temp.push(c);
        } else if (c == ')') {
            while (!temp.empty() && temp.top() != '(') {
                res += temp.top();
                temp.pop();
            }
            if (!temp.empty()) {
                temp.pop();
            }
        } else {
            while (!temp.empty() && op(temp.top()) >= op(c)) {
                res += temp.top();
                temp.pop();
            }
            temp.push(c);
        }
    }
    if (!temp.empty()) {
        while (!temp.empty()) {
            res += temp.top();
            temp.pop();
        }
    }
    printstr(res);
    return 0;
}