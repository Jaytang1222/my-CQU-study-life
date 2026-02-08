#include<iostream>
#include<vector>
#include<algorithm>

using namespace std;

void merge(vector<int>& input, int l, int m, int r) {
    vector<int> left(input.begin() + l, input.begin() + m + 1);
    vector<int> right(input.begin() + m + 1, input.begin() + r + 1);
    int n1 = left.size(), n2 = right.size();
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (left[i] < right[j]) {
            input[k] = left[i];
            k++;
            i++;
        } else {
            input[k] = right[j];
            k++;
            j++;
        }
    }
    while (i < n1) {
        input[k] = left[i];
        k++;
        i++;
    }
    while (j < n2) {
        input[k] = right[j];
        k++;
        j++;
    }
}

void mergesort(vector<int>& input, int l, int r) {
    if (l < r) {
        int m = l + ((r - l) >> 1);
        mergesort(input, l, m);
        mergesort(input, m + 1, r);
        merge(input, l, m, r);
    }
}

int main() {
    vector<int> input = {12, 21, 13, 25, 16, 27};
    mergesort(input, 0, 5);
    for (int i = 0; i < 6; i++) {
        cout << input[i] << " ";
    }
    return 0;
}