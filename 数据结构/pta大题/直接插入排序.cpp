#include<iostream>
#include<vector>

using namespace std;

void printvec(vector<int>& v) {
    for (int num : v) {
        cout << num << " ";
    }
    cout << endl;
} 

void insertsort(vector<int>& v) {
    int n = v.size();
    for (int i = 1; i < n; i++) {
        int key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
        printvec(v);
    }
    
}

int main() {
    int n;
    while (cin >> n) {
        int temp;
        vector<int> v;
        for (int i = 0; i < n; i++) {
            cin >> temp;
            v.push_back(temp);
        }
        insertsort(v);
    }
    return 0;
}