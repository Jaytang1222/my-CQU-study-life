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

BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*> objects)
{
    BVHBuildNode* node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    if (objects.size() == 1) {
        // Create leaf _BVHBuildNode_
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
        Bounds3 centroidBounds;
        for (int i = 0; i < objects.size(); ++i)
            centroidBounds =
                Union(centroidBounds, objects[i]->getBounds().Centroid());

        int dim = centroidBounds.maxExtent();

        // If using SAH split method
        if (splitMethod == SplitMethod::SAH) {
            const int nBuckets = 12;
            struct BucketInfo { int count = 0; Bounds3 bounds; };
            BucketInfo buckets[nBuckets];

            auto extent = centroidBounds.Diagonal();
            float minCoord = (dim == 0 ? centroidBounds.pMin.x : (dim == 1 ? centroidBounds.pMin.y : centroidBounds.pMin.z));
            float invExtent = 1.0f / (dim == 0 ? extent.x : (dim == 1 ? extent.y : extent.z));
            bool degenerateAxis = ((dim == 0 ? extent.x : (dim == 1 ? extent.y : extent.z)) <= 0);

            if (!degenerateAxis) {
                for (auto obj : objects) {
                    Vector3f c = obj->getBounds().Centroid();
                    float coord = (dim == 0 ? c.x : (dim == 1 ? c.y : c.z));
                    int b = (int)((coord - minCoord) * invExtent * nBuckets);
                    if (b == nBuckets) b = nBuckets - 1;
                    buckets[b].count++;
                    buckets[b].bounds = Union(buckets[b].bounds, obj->getBounds());
                }

                float cost[nBuckets - 1];
                Bounds3 leftBounds[nBuckets - 1];
                Bounds3 rightBounds[nBuckets - 1];
                int leftCount[nBuckets - 1];
                int rightCount[nBuckets - 1];

                // Prefix for left
                Bounds3 b;
                int count = 0;
                for (int i = 0; i < nBuckets - 1; ++i) {
                    b = Union(b, buckets[i].bounds);
                    leftBounds[i] = b;
                    count += buckets[i].count;
                    leftCount[i] = count;
                }
                // Suffix for right
                b = Bounds3();
                count = 0;
                for (int i = nBuckets - 1; i > 0; --i) {
                    b = Union(b, buckets[i].bounds);
                    if (i - 1 < nBuckets - 1) rightBounds[i - 1] = b;
                    count += buckets[i].count;
                    if (i - 1 < nBuckets - 1) rightCount[i - 1] = count;
                }

                float parentArea = bounds.SurfaceArea();
                int bestSplit = -1;
                float bestCost = std::numeric_limits<float>::infinity();
                for (int i = 0; i < nBuckets - 1; ++i) {
                    float c = 1.0f + (leftCount[i] * leftBounds[i].SurfaceArea() +
                                       rightCount[i] * rightBounds[i].SurfaceArea()) / parentArea;
                    cost[i] = c;
                    if (c < bestCost) { bestCost = c; bestSplit = i; }
                }

                // If SAH not worth splitting, create leaf
                if (objects.size() <= (size_t)maxPrimsInNode || bestSplit == -1) {
                    node->left = recursiveBuild(std::vector<Object*>{objects.begin(), objects.begin() + 1});
                    node->right = recursiveBuild(std::vector<Object*>{objects.begin() + 1, objects.end()});
                } else {
                    auto midPredicate = [&](Object* obj) {
                        Vector3f c = obj->getBounds().Centroid();
                        float coord = (dim == 0 ? c.x : (dim == 1 ? c.y : c.z));
                        int bidx = (int)((coord - minCoord) * invExtent * nBuckets);
                        if (bidx == nBuckets) bidx = nBuckets - 1;
                        return bidx <= bestSplit;
                    };
                    auto mid = std::partition(objects.begin(), objects.end(), midPredicate);
                    if (mid == objects.begin() || mid == objects.end()) {
                        // fallback to equal split to avoid degenerate partition
                        std::sort(objects.begin(), objects.end(), [dim](auto f1, auto f2){
                            auto c1 = f1->getBounds().Centroid();
                            auto c2 = f2->getBounds().Centroid();
                            return (dim==0?c1.x:(dim==1?c1.y:c1.z)) < (dim==0?c2.x:(dim==1?c2.y:c2.z));
                        });
                        mid = objects.begin() + (objects.size()/2);
                    }
                    std::vector<Object*> leftshapes(objects.begin(), mid);
                    std::vector<Object*> rightshapes(mid, objects.end());
                    node->left = recursiveBuild(leftshapes);
                    node->right = recursiveBuild(rightshapes);
                }
                node->bounds = Union(node->left->bounds, node->right->bounds);
                return node;
            }
        }

        // NAIVE split (fallback and for SplitMethod::NAIVE)
        std::sort(objects.begin(), objects.end(), [dim](auto f1, auto f2) {
            return (dim==0?f1->getBounds().Centroid().x:(dim==1?f1->getBounds().Centroid().y:f1->getBounds().Centroid().z)) <
                   (dim==0?f2->getBounds().Centroid().x:(dim==1?f2->getBounds().Centroid().y:f2->getBounds().Centroid().z));
        });

        auto beginning = objects.begin();
        auto middling = objects.begin() + (objects.size() / 2);
        auto ending = objects.end();

        auto leftshapes = std::vector<Object*>(beginning, middling);
        auto rightshapes = std::vector<Object*>(middling, ending);

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));

        node->left = recursiveBuild(leftshapes);
        node->right = recursiveBuild(rightshapes);

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
    if (node == nullptr) return isect;

    Vector3f invDir = ray.direction_inv;
    std::array<int, 3> dirIsNeg = { ray.direction.x < 0, ray.direction.y < 0, ray.direction.z < 0 };

    if (!node->bounds.IntersectP(ray, invDir, dirIsNeg))
        return isect;

    if (node->object != nullptr)
    {
        return node->object->getIntersection(ray);
    }

    Intersection hitLeft = getIntersection(node->left, ray);
    Intersection hitRight = getIntersection(node->right, ray);

    if (hitLeft.happened && hitRight.happened)
        return (hitLeft.distance < hitRight.distance) ? hitLeft : hitRight;
    else if (hitLeft.happened)
        return hitLeft;
    else
        return hitRight;
}