#include<iostream>

using namespace std;

int res = 1;

int recursion(int n) {
    if (n == 1) return res;
    res = res * n;
    return recursion(n - 1);
}

int main() {
    int n;
    cin >> n;
    cout << recursion(n);
    return 0;
}