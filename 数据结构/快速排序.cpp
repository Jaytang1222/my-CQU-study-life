void swap(int* a,int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}
void quickSort(int* a,int begin,int end)
{
    if(begin >= end) return;//当begin>=end，结束递归操作
    int L = begin,R = end,key = begin;//定义L,R,key下标
    while(L < R)
    {
        while(a[R] >= a[key] && L < R)//右边先走，右边大于等于key就一直走下去，否则停下来，加上判断L < R，防止越界
        {
            R--;
        }
        while(a[L] <= a[key] && L < R)//同理
        {
            L++; 
        }
        swap(&a[L],&a[R]);//都停下来后交换
    }
    swap(&a[key],&a[R]);//最后key值和相遇点交换
    key = R;//以相遇点作为分界点，递归操作
    quickSort(a,begin,key - 1);//递归
    quickSort(a,key + 1,end);
}

//写法二：
#include <iostream>
#include <vector>
using namespace std;

// 快速排序的Partition过程
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[low]; // 选择第一个元素作为基准元素
    int left = low + 1;
    int right = high;

    while (true) {
        while (left <= right && arr[left] <= pivot) {
            left++;
        }

        while (left <= right && arr[right] > pivot) {
            right--;
        }

        if (left <= right) {
            swap(arr[left], arr[right]);
        } else {
            break;
        }
    }

    swap(arr[low], arr[right]);
    return right;
}

// 快速排序递归函数
void quickSort(vector<int>& arr, int low, int high) {
    if (low <= high) {
        // 进行一趟Partition
        int pivotIndex = partition(arr, low, high);

        // 输出当前趟的结果
        for (int i = 0; i < arr.size(); ++i) {
            cout << arr[i] << " ";
        }
        cout << endl;

        // 递归对基准元素左右两侧进行排序
        quickSort(arr, low, pivotIndex - 1);
        quickSort(arr, pivotIndex + 1, high);
    }
}

int main() {
    int n;
    cin >> n;
    vector<int> arr(n);
    for (int i = 0; i < n; ++i) {
        cin >> arr[i];
    }
    // 调用快速排序函数
    quickSort(arr, 0, n - 1);
    return 0;
}