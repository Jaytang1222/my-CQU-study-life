#include<iostream>
#include<vector>

using namespace std;

void printvec(vector<int>& input) {
    for (int num : input) {
        cout << num << " ";
    }
    cout << endl;
}

void insertsort(vector<int>& input) {
    int n = input.size();
    for (int i = 1; i < n; i++) {
        int key = input[i];
        int index = 0;
        for (int j = i - 1; j >= 0; j--) {
            if (input[j] > key) {
                input[j + 1] = input[j];
            } else {
                index = j + 1;
                break;
            }
        }
        input[index] = key;
        printvec(input);
    }
}

int main() {
    int n;
    while (cin >> n) {
        vector<int> input;
        for (int i = 0; i < n ;i++) {
            int temp;
            cin >> temp;
            input.push_back(temp);
        }
        insertsort(input);
    }
    return 0;
}