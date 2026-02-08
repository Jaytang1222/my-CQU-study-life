#include <iostream>
#include <vector>
#include <algorithm>

/**
 * @brief 核心合并函数：将 arr[left...mid] 和 arr[mid+1...right] 两个有序子数组合并。
 * @param arr 待排序的数组。
 * @param left 第一个子数组的起始索引。
 * @param mid 第一个子数组的结束索引。
 * @param right 第二个子数组的结束索引。
 */
void merge(std::vector<int>& arr, int left, int mid, int right) {
    // 计算左右子数组的长度
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // 创建临时数组来存储左右子数组
    // 注意：在实际应用中，为了节省反复分配/释放内存，通常只在 mergeSort 外分配一次辅助空间
    std::vector<int> L(n1);
    std::vector<int> R(n2);

    // 复制数据到临时数组 L 和 R
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    // --- 开始合并 ---
    int i = 0; // 临时数组 L 的当前索引
    int j = 0; // 临时数组 R 的当前索引
    int k = left; // 主数组 arr 的当前写入索引

    // 比较 L 和 R 中的元素，将较小的放入 arr 中
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // 将 L 中剩余的元素（如果有）复制到 arr
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // 将 R 中剩余的元素（如果有）复制到 arr
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}


/**
 * @brief 归并排序主函数：递归地分割数组。
 * @param arr 待排序的数组。
 * @param left 当前子数组的起始索引。
 * @param right 当前子数组的结束索引。
 */
void mergeSort(std::vector<int>& arr, int left, int right) {
    // 递归终止条件：当子数组只有一个元素时 (left >= right)，视为有序
    if (left < right) {
        // 1. 分割 (Divide): 找到中点
        int mid = left + (right - left) / 2;

        // 2. 征服 (Conquer): 递归排序左右两半
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        // 3. 合并 (Combine): 将排序好的两半合并
        merge(arr, left, mid, right);
    }
}


// --- 示例用法 ---
int main() {
    std::vector<int> data = {12, 11, 13, 5, 6, 7};
    int n = data.size();

    std::cout << "原始数组: ";
    for (int x : data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // 调用归并排序
    mergeSort(data, 0, n - 1);

    std::cout << "排序结果: ";
    for (int x : data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    
    return 0;
}