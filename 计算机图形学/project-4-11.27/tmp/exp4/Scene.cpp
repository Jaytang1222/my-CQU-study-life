#include "Scene.hpp"


void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
        const Ray &ray,
        const std::vector<Object*> &objects,
        float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }


    return (*hitObject != nullptr);
}

Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    // 求当前相机射线与场景的交点
    Intersection p_inter = intersect(ray);
    if (!p_inter.happened) {
        return Vector3f();
    }
    // 若命中光源，直接返回其发射的辐射
    if (p_inter.m->hasEmission()) {
        return p_inter.m->getEmission();
    }

    float EPLISON = 0.0001;
    Vector3f l_dir;   // 直接光照
    Vector3f l_indir; // 间接光照
    
    // 在所有发光体上按面积采样一个点
    Intersection x_inter;
    float pdf_light = 0.0f;
    sampleLight(x_inter, pdf_light);    
    
    // 从当前交点出发指向光源上的采样点
    Vector3f p = p_inter.coords;
    Vector3f x = x_inter.coords;
    Vector3f ws_dir = (x - p).normalized();
    float ws_distance = (x - p).norm();
    Vector3f N = p_inter.normal.normalized();
    Vector3f NN = x_inter.normal.normalized();
    Vector3f emit = x_inter.emit;

    // 构建阴影射线检测遮挡
    Ray ws_ray(p, ws_dir); 
    Intersection ws_ray_inter = intersect(ws_ray);
    // 若阴影射线未被遮挡，则累加直接光
    if(ws_ray_inter.distance - ws_distance > -EPLISON) {
        l_dir = emit * p_inter.m->eval(ray.direction, ws_ray.direction, N) 
            * dotProduct(ws_ray.direction, N)
            * dotProduct(-ws_ray.direction, NN)
            / std::pow(ws_distance, 2)
            / pdf_light;
    }
    
    // 以 Russian Roulette 概率决定是否继续递归
    if(get_random_float() > RussianRoulette) {
        return l_dir;
    }

    l_indir = 0.0;

    // 依据材质的 BRDF 采样新的入射方向
    Vector3f wi_dir = p_inter.m->sample(ray.direction, N).normalized();
    Ray wi_ray(p_inter.coords, wi_dir);
    // 若命中的不是发光体，继续递归累计间接光
    Intersection wi_inter = intersect(wi_ray);
    if (wi_inter.happened && (!wi_inter.m->hasEmission())) {
        l_indir = castRay(wi_ray, depth + 1) * p_inter.m->eval(ray.direction, wi_ray.direction, N)
            * dotProduct(wi_ray.direction, N)
            / p_inter.m->pdf(ray.direction, wi_ray.direction, N)
            / RussianRoulette;
    }
    
    return l_dir + l_indir;
}