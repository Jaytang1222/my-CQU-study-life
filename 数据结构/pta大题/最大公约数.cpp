#include<iostream>
#include<algorithm>

using namespace std;

long long getgcd(long long a, long long  b) {
    while ( b != 0) {
        long long temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

int main() {
    long long m, n;
    cin >> m >> n;
    long long gcdnum = getgcd(m, n);
    cout << gcdnum << '\n';
    return 0;
}