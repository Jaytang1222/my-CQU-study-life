#include<iostream>
#include<stack>

using namespace std;

int main() {
    int n;
    cin>>n;
    for (int i = 0; i < n; i++) {
        stack<char> st;
        string input;
        cin>>input;
        for (char c : input) {
            if (c == '{') {
                if (st.empty() || st.top() == '}') {
                    st.push('}');
                } else {
                    break;
                }
            } else if (c == '[') {
                if (st.empty() || st.top() == '}' || st.top() == ']') {
                    st.push(']');
                } else {
                    break;
                }
            } else if (c == '(') {
                if (st.empty() || st.top() != '>') {
                    st.push(')');
                } else {
                    break;
                }
            } else if (c == '<') {
                st.push('>');
            } else if (!st.empty() && c == st.top()) {
                st.pop();
            }
        }
        cout<<(st.empty() ? "YES" : "NO")<<endl;
    }
    return 0;
}