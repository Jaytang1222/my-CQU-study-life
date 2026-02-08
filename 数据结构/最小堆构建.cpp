#include <iostream>
#include <vector>
#include <algorithm> // 仅用于swap交换元素

using namespace std;

// 小顶堆类（极简版）
class MinHeap {
private:
    vector<int> heap; // 底层数组存储堆

    // 向上调整：插入后修复堆性质（核心函数）
    void siftUp(int idx) {
        // 未到堆顶时循环调整
        while (idx > 0) {
            int parentIdx = (idx - 1) / 2; // 计算父节点下标
            // 小顶堆：子节点 < 父节点 → 交换，继续向上
            if (heap[idx] < heap[parentIdx]) {
                swap(heap[idx], heap[parentIdx]);
                idx = parentIdx; // 移动到父节点位置，继续检查
            } else {
                break; // 满足小顶堆性质，停止调整
            }
        }
    }

public:
    // 插入单个元素（核心接口）
    void insert(int num) {
        heap.push_back(num);       // 步骤1：新元素插入到数组末尾
        siftUp(heap.size() - 1);   // 步骤2：从末尾向上调整
    }

    // 批量插入：按顺序插入一系列数字
    void insertBatch(const vector<int>& nums) {
        for (int num : nums) {
            insert(num);
        }
    }

    // 打印堆（验证结果）
    void print() {
        for (size_t i = 0; i < heap.size(); ++i) {
            cout << heap[i] << (i == heap.size() - 1 ? "\n" : " ");
        }
    }

    // 获取堆顶（最小元素）
    int getTop() {
        return heap.empty() ? -1 : heap[0];
    }
};

// 测试示例
int main() {
    // 待插入的数字序列
    vector<int> nums = {7, 5, 8, 4, 2, 3, 6, 1};
    MinHeap minHeap;

    // 顺序插入所有数字
    cout << "插入顺序：7 5 8 4 2 3 6 1\n";
    minHeap.insertBatch(nums);

    // 输出结果
    cout << "小顶堆最终数组：";
    minHeap.print();
    cout << "小顶堆顶（最小值）：" << minHeap.getTop() << endl;

    return 0;
}