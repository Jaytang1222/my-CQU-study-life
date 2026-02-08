#include<iostream>
#include<vector>
#include<algorithm>

using namespace std;

void printvec(vector<int>& input) {
    for (int num : input) {
        cout << num << " ";
    }
    cout << endl;
}

void boobsort(vector<int>& input, int n) {
    for (int i = 0; i < 3; i++) {
        for (int j =  n - 1; j > i; j--) {
            if (input[j] < input[j - 1]) {
                swap(input[j], input[j - 1]);
            }
        }
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
    boobsort(input, n);
    return 0;
}