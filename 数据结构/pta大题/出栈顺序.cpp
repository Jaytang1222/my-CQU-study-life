#include<iostream>
#include<stack>
#include<algorithm>
using namespace std;

int main() {
    string a, b;
    cin >> a >> b;
    stack<char> temp;
    //int startin = 0;
    int start = 0;
    int size = a.size();
    string res;
    for (int i = 0; i < size; i++) {
        temp.push(a[i]);
        res += 'P';
        while (!temp.empty() && temp.top() == b[start]) {
            temp.pop();
            res += 'O';
            start++;
        }
    }
    if (temp.empty()) {
        cout << "right" << '\n';
        cout << res;
    } else {
        cout << "wrong" << '\n';
        string remaining;
        while (!temp.empty()) {
            remaining += temp.top();
            temp.pop();
        }
        reverse(remaining.begin(), remaining.end());
        cout << remaining;
    }
    return 0;
}