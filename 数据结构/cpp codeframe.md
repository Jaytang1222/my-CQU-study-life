1、十进制转二进制:
string tenT0two (int n) {
    if (n == 0) return "0";
    string bin;
    while (n > 0) {
        bin += (n & 1) ? '1' : '0';
        n >>= 1;
    }
    reverse(bin.begin(), bin.end());
    return bin;
}

2、getline函数:
// 示例 2: 读取到逗号 ,
    std::cout << "Enter data separated by comma: ";
    // 假设用户输入: apple,banana,cherry
    std::getline(std::cin, line, ',');
    std::cout << "Read before comma: " << line << "\n"; // 输出 "Read before comma: apple"

3、位运算的应用:
-快速乘除计算-左移一位相当于乘以 2： x << n 相当于 x×2 n
例如：5 << 1=10
右移一位相当于除以 2 (取整数部分)： x >> n 相当于 ⌊x/2 n ⌋（对于无符号数或正有符号数）。
例如：5 >> 1=2

-一个整数 x 是奇数还是偶数-
取决于它的最低位（2^0位）是 1 还是 0
偶数：if (x & 1 == 0)
奇数：if (x & 1 == 1)

3、优先队列
大顶堆：priority_queue<int> maxHeap;
小顶堆：priority_queue<int, vector<int>, greater<int>> minHeap;

