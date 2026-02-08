#include <algorithm>
#include <cassert>
#include "BVH.hpp"

BVHAccel::BVHAccel(std::vector<Object*> p, int maxPrimsInNode,
                   SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), splitMethod(splitMethod),
      primitives(std::move(p))
{
    time_t start, stop;
    time(&start);
    if (primitives.empty())
        return;

    root = recursiveBuild(primitives);

    time(&stop);
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    printf(
        "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
        hrs, mins, secs);
}

float computeSAHCost(const std::vector<Object*>& objects, int splitIndex, int dim) {
    // 分割为左右两组
    auto left = std::vector<Object*>(objects.begin(), objects.begin() + splitIndex);
    auto right = std::vector<Object*>(objects.begin() + splitIndex, objects.end());
    
    // 计算左右包围盒
    Bounds3 leftBounds, rightBounds;
    for (auto obj : left) leftBounds = Union(leftBounds, obj->getBounds());
    for (auto obj : right) rightBounds = Union(rightBounds, obj->getBounds());
    
    // 计算表面积
    float leftArea = leftBounds.SurfaceArea();
    float rightArea = rightBounds.SurfaceArea();
    float totalArea = leftArea + rightArea;
    
    if (totalArea < 1e-9f) return 0.0f; // 避免除以零
    
    // SAH代价公式：C = (leftArea / totalArea) * leftCount + (rightArea / totalArea) * rightCount
    // 忽略遍历代价（C_travel），仅考虑相交测试代价
    return (leftArea * left.size() + rightArea * right.size()) / totalArea;
}

BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*> objects)
{
    BVHBuildNode* node = new BVHBuildNode();
    // 计算所有物体的包围盒
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    if (objects.size() == 1) {
        // 叶子节点
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }
    else if (objects.size() == 2) {
        node->left = recursiveBuild(std::vector{objects[0]});
        node->right = recursiveBuild(std::vector{objects[1]});
        node->bounds = Union(node->left->bounds, node->right->bounds);
        return node;
    }
    else {
        // SAH分割：找最优分割轴和分割点
        Bounds3 centroidBounds;
        for (auto obj : objects) centroidBounds = Union(centroidBounds, obj->getBounds().Centroid());
        int bestDim = centroidBounds.maxExtent();  // 最优分割轴
        std::sort(objects.begin(), objects.end(), [bestDim](auto f1, auto f2) {
            return f1->getBounds().Centroid()[bestDim] < f2->getBounds().Centroid()[bestDim];
        });
        
        // 遍历所有可能的分割点，找代价最小的
        float minCost = std::numeric_limits<float>::max();
        int bestSplitIndex = objects.size() / 2;  // 默认中间分割
        for (int i = 1; i < objects.size(); ++i) {
            float cost = computeSAHCost(objects, i, bestDim);
            if (cost < minCost) {
                minCost = cost;
                bestSplitIndex = i;
            }
        }
        
        // 按最优分割点分割
        auto leftShapes = std::vector<Object*>(objects.begin(), objects.begin() + bestSplitIndex);
        auto rightShapes = std::vector<Object*>(objects.begin() + bestSplitIndex, objects.end());
        assert(objects.size() == leftShapes.size() + rightShapes.size());
        
        node->left = recursiveBuild(leftShapes);
        node->right = recursiveBuild(rightShapes);
        node->bounds = Union(node->left->bounds, node->right->bounds);
    }
    return node;
}

Intersection BVHAccel::Intersect(const Ray& ray) const
{
    Intersection isect;
    if (!root)
        return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

Intersection BVHAccel::getIntersection(BVHBuildNode* node, const Ray& ray) const
{
    // TODO Traverse the BVH to find intersection
     Intersection isect;
    // 1. 先判断光线是否与当前节点的包围盒相交，不相交则直接返回空
    Vector3f invDir = Vector3f(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
    std::array<int, 3> dirIsNeg = {ray.direction.x > 0, ray.direction.y > 0, ray.direction.z > 0};
    if (!node->bounds.IntersectP(ray, invDir, dirIsNeg))
        return isect;
    
    // 2. 如果是叶子节点（有物体），直接计算光线与物体的交点
    if (node->left == nullptr && node->right == nullptr) {
        isect = node->object->getIntersection(ray);
        return isect;
    }
    
    // 3. 非叶子节点，递归遍历左右子节点
    Intersection leftIsect = getIntersection(node->left, ray);
    Intersection rightIsect = getIntersection(node->right, ray);
    
    // 4. 返回距离光线最近的交点
    return leftIsect.distance < rightIsect.distance ? leftIsect : rightIsect;
}