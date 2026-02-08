#include<iostream>
#include<vector>
#include<algorithm>
#include<numeric>

using namespace std;

int main() {
    int n;
    cin >> n;
    vector<int> input(n);
    for (int i = 0; i < n; i++) {
        cin >> input[i];
    }
    vector<int> dp(input.begin(), input.end());
    for (int i = 1; i < n; i++) {
        dp[i] = max(dp[i], dp[i] + dp[i - 1]);
    }
    int res = *max_element(dp.begin(), dp.end());
    if (res < 0) {
        cout << 0 << '\n';
        return 0;
    }
    cout << res << '\n';
    return 0;
}