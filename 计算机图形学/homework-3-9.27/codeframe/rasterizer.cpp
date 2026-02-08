// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(int x, int y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    // 计算三个边的向量
    Eigen::Vector2f v0(_v[0].x(), _v[0].y());
    Eigen::Vector2f v1(_v[1].x(), _v[1].y());
    Eigen::Vector2f v2(_v[2].x(), _v[2].y());
    Eigen::Vector2f p(x, y);

    // 计算三个叉积
    Eigen::Vector2f e0 = v1 - v0;
    Eigen::Vector2f e1 = v2 - v1;
    Eigen::Vector2f e2 = v0 - v2;

    Eigen::Vector2f c0 = p - v0;
    Eigen::Vector2f c1 = p - v1;
    Eigen::Vector2f c2 = p - v2;

    float cross0 = e0.x() * c0.y() - e0.y() * c0.x();
    float cross1 = e1.x() * c1.y() - e1.y() * c1.x();
    float cross2 = e2.x() * c2.y() - e2.y() * c2.x();

    // 检查所有叉积是否同号（都为正或都为负）
    return (cross0 > 0 && cross1 > 0 && cross2 > 0) || (cross0 < 0 && cross1 < 0 && cross2 < 0);
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    // 1. 创建三角形的2D bounding box
    // 初始化边界框坐标为顶点的极值
    float min_x = std::min({v[0].x(), v[1].x(), v[2].x()});
    float max_x = std::max({v[0].x(), v[1].x(), v[2].x()});
    float min_y = std::min({v[0].y(), v[1].y(), v[2].y()});
    float max_y = std::max({v[0].y(), v[1].y(), v[2].y()});
    
    // 将边界框限制在屏幕范围内
    min_x = std::max(0.0f, min_x);
    max_x = std::min((float)width - 1, max_x);
    min_y = std::max(0.0f, min_y);
    max_y = std::min((float)height - 1, max_y);

    // 2. 遍历bounding box内的所有像素
    for (int y = static_cast<int>(min_y); y <= static_cast<int>(max_y); ++y) {
        for (int x = static_cast<int>(min_x); x <= static_cast<int>(max_x); ++x) {
            // 计算像素中心坐标（屏幕空间）
            float px = x + 0.5f;  // 像素中心x坐标
            float py = y + 0.5f;  // 像素中心y坐标
            
            // 检查像素中心是否在三角形内
            if (insideTriangle(px, py, t.v)) {
                // 计算重心坐标用于插值
                auto [alpha, beta, gamma] = computeBarycentric2D(px, py, t.v);
                
                // 3. 计算插值的深度值（考虑透视矫正）
                float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = (alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w()) * w_reciprocal;
                
                // 获取当前像素在深度缓冲区中的索引
                int idx = get_index(x, y);
                
                // 4. 深度测试：如果当前点更靠近相机（z值更小）
                if (z_interpolated < depth_buf[idx]) {
                    // 更新深度缓冲区
                    depth_buf[idx] = z_interpolated;
                    // 设置像素颜色
                    Eigen::Vector3f color = t.getColor();
                    set_pixel(Eigen::Vector3f(x, y, z_interpolated), color);
                }
            }
        }
    }
    // If so, use the following code to get the interpolated z value.
    //auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    //float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    //float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    //z_interpolated *= w_reciprocal;

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on