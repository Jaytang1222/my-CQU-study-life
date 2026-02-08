#include<iostream>
#include<vector>
#include<algorithm>
#include<functional>

using namespace std;

void printvec(vector<int>& input) {
    int n = input.size();
    for (int num : input) {
        cout << num << " ";
    }
    cout <<endl;
}

int main() {
    int n;
    cin >> n;
    vector<int> input(n);
    for (int i = 0; i < n; i++) {
        cin >> input[i];
    }
    vector<int> temp = input;
    make_heap(input.begin(), input.end());
    make_heap(temp.begin(), temp.end(), greater<int>());
    printvec(input);
    printvec(temp);
    return 0;
}