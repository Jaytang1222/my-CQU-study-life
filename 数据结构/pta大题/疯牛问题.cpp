#include<iostream>
#include<vector>
#include<algorithm>
#include<numeric>

using namespace std;

bool isok(vector<int>& input, int m, int mid) {
    int temp = 0;
    int count = 1;
    for (int i = 0; i < input.size(); i++) {
        if (count > m) return false;
        if (temp + input[i] <= mid) {
            temp += input[i];
        } else {
            count++;
            temp = input[i];
        }
    }
    return count <= m;
}



int main() {
    int n, m;
    cin >> n >> m;
    vector<int> input(n);
    for (int i = 0; i < n; i++) {
        cin >> input[i];
    }
    int low = *max_element(input.begin(), input.end());
    int high = accumulate(input.begin(), input.end(), 0);
    int res = high;
    while (low <= high) {
        int mid = low + ((high - low) >> 1);
        if (isok(input, m, mid)) {
            res = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    cout << res << endl;
    return 0;
}