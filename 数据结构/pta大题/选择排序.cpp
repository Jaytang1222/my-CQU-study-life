#include<iostream>
#include<vector>
#include<algorithm>

using namespace std;

void printvec(vector<int>& input) {
    for (int num : input ) {
        cout << num << " ";
    }
    cout << endl;
}

void choosesort(vector<int>& input, int n) {
    int start = n - 1;
    for (int i = start; i > start - 3; i--) {
        int maxindex = 0;
        int maxnum = INT_MIN;
        for (int j = i; j >= 0; j--) {
            if (input[j] > maxnum) {
                maxnum = input[j];
                maxindex = j;
            }
        }
        swap(input[i], input[maxindex]);
        printvec(input);
    }
}

int main() {
    int n;
    cin >> n;
    vector<int> input(n);
    for (int i = 0; i < n; i++) {
        cin >> input[i];
    }
    choosesort(input, n);
    return 0;
}