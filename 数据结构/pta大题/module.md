#include<bits/stdc++.h>

sort(edges.begin(), edges.end(), [](const edge& a, const edge& b) {
    return a.val < b.val;
})