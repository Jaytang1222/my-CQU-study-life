#include <iostream>
#include <opencv2/opencv.hpp>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    // TODO: Use the same projection matrix from the previous assignments

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    float fov_rad = eye_fov * M_PI / 180.0f;
    float tan_half_fov = tan(fov_rad / 2.0f);

    projection(0, 0) = 1.0f / (tan_half_fov * aspect_ratio);
    
    projection(1, 1) = 1.0f / tan_half_fov;

    projection(2, 2) = (zFar + zNear) / (zNear - zFar);
    projection(2, 3) = 2.0f * zFar * zNear / (zNear - zFar);
    
    projection(3, 2) = -1.0f;
    projection(3, 3) = 0.0f;
    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        // TODO: Get the texture value at the texture coordinates of the current fragment

    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.

         // 计算光源到片段的方向向量
        Eigen::Vector3f light_dir = light.position - point;
        float distance = light_dir.norm();  // 光源与片段的距离
        Eigen::Vector3f L = light_dir.normalized();  // 归一化光源方向

        // 计算视线方向（片段到相机）
        Eigen::Vector3f V = (eye_pos - point).normalized();

        // 计算半程向量（Blinn-Phong模型）
        Eigen::Vector3f H = (L + V).normalized();

        // 计算衰减因子（1 / r²）
        float attenuation = 1.0f / (distance * distance);

        // 环境光分量
        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);

        // 漫反射分量（使用纹理颜色作为kd）
        float diff_dot = std::max(normal.dot(L), 0.0f);
        Eigen::Vector3f diffuse = kd.cwiseProduct(light.intensity) * diff_dot * attenuation;

        // 高光分量
        float spec_dot = std::max(normal.dot(H), 0.0f);
        Eigen::Vector3f specular = ks.cwiseProduct(light.intensity) * std::pow(spec_dot, p) * attenuation;

        // 累加当前光源的贡献
        result_color += ambient + diffuse + specular;
    }

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        
        // 计算光源到片段的方向向量
        Eigen::Vector3f light_dir = light.position - point;
        float distance = light_dir.norm();  // 光源到片段的距离
        Eigen::Vector3f L = light_dir.normalized();  // 归一化光源方向

        // 计算视线方向（片段到相机）
        Eigen::Vector3f V = (eye_pos - point).normalized();

        // 计算半程向量（Blinn-Phong模型核心）
        Eigen::Vector3f H = (L + V).normalized();

        // 计算衰减因子（1 / r²）
        float attenuation = 1.0f / (distance * distance);

        // 环境光分量
        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);

        // 漫反射分量
        float diff_dot = std::max(normal.dot(L), 0.0f);
        Eigen::Vector3f diffuse = kd.cwiseProduct(light.intensity) * diff_dot * attenuation;

        // 高光分量
        float spec_dot = std::max(normal.dot(H), 0.0f);
        Eigen::Vector3f specular = ks.cwiseProduct(light.intensity) * std::pow(spec_dot, p) * attenuation;

        // 累加当前光源的贡献
        result_color += ambient + diffuse + specular;
    }

    return result_color * 255.f;
}



Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;
    
    // TODO: Implement displacement mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)

    // 实现位移映射
    // 1. 计算切线向量t和副切线向量b，构建TBN矩阵
    float x = normal.x();
    float y = normal.y();
    float z = normal.z();
    
    // 计算切线向量t（按注释公式）
    float denom = sqrt(x*x + z*z) + 1e-8;  // 避免除零
    Eigen::Vector3f t(
        x*y / denom,
        denom,
        z*y / denom
    );
    t.normalize();
    
    // 计算副切线向量b（法向量与切线的叉积）
    Eigen::Vector3f b = normal.cross(t).normalized();
    
    // 构建TBN矩阵
    Eigen::Matrix3f TBN;
    TBN.col(0) = t;
    TBN.col(1) = b;
    TBN.col(2) = normal;

    // 2. 从纹理获取高度值并计算位移
    float u = payload.tex_coords.x();
    float v = payload.tex_coords.y();
    float h = 0.0f;
    
    if (payload.texture) {
        // 获取当前纹理坐标的高度（使用纹理亮度）
        h = payload.texture->getColor(u, v).norm();
        
        // 位移顶点位置（按注释公式）
        point = point + kn * normal * h;
    }

    // 3. 计算高度梯度和扰动法向量
    float h_u = payload.texture ? payload.texture->getColor(u + 1e-4, v).norm() : h;
    float h_v = payload.texture ? payload.texture->getColor(u, v + 1e-4).norm() : h;
    
    // 计算dU和dV（按注释公式）
    float dU = kh * kn * (h_u - h);
    float dV = kh * kn * (h_v - h);
    
    // 计算切线空间法向量并转换到世界空间
    Eigen::Vector3f ln(-dU, -dV, 1.0f);
    Eigen::Vector3f n = (TBN * ln).normalized();  // 扰动后的法向量


    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.

        // 光源方向向量（基于位移后的位置）
        Eigen::Vector3f light_dir = light.position - point;
        float distance = light_dir.norm();
        Eigen::Vector3f L = light_dir.normalized();
        
        // 视线方向向量（基于位移后的位置）
        Eigen::Vector3f V = (eye_pos - point).normalized();
        
        // 半程向量
        Eigen::Vector3f H = (L + V).normalized();
        
        // 衰减因子
        float attenuation = 1.0f / (distance * distance);
        
        // 环境光分量
        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);
        
        // 漫反射分量（使用扰动后的法向量）
        float diff_dot = std::max(n.dot(L), 0.0f);
        Eigen::Vector3f diffuse = kd.cwiseProduct(light.intensity) * diff_dot * attenuation;
        
        // 高光分量（使用扰动后的法向量）
        float spec_dot = std::max(n.dot(H), 0.0f);
        Eigen::Vector3f specular = ks.cwiseProduct(light.intensity) * std::pow(spec_dot, p) * attenuation;
        
        // 累加光源贡献
        result_color += ambient + diffuse + specular;

    }

    return result_color * 255.f;
}


Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;


    float kh = 0.2, kn = 0.1;

    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)

    // 实现Bump Mapping
    // 1. 计算切线向量t和副切线向量b，构建TBN矩阵
    float x = normal.x();
    float y = normal.y();
    float z = normal.z();
    
    // 计算切线向量t（按注释公式）
    float denom = sqrt(x*x + z*z) + 1e-8;  // 避免除零
    Eigen::Vector3f t(
        x*y / denom,
        denom,
        z*y / denom
    );
    t.normalize();  // 归一化切线向量
    
    // 计算副切线向量b（法向量与切线的叉积）
    Eigen::Vector3f b = normal.cross(t).normalized();
    
    // 构建TBN矩阵（切线空间到世界空间的转换矩阵）
    Eigen::Matrix3f TBN;
    TBN.col(0) = t;
    TBN.col(1) = b;
    TBN.col(2) = normal;

    // 2. 从纹理获取高度信息并计算梯度
    float u = payload.tex_coords.x();
    float v = payload.tex_coords.y();
    
    // 采样当前和相邻纹理坐标的高度（使用纹理亮度作为高度值）
    float h = payload.texture ? payload.texture->getColor(u, v).norm() : 0.0f;
    float h_u = payload.texture ? payload.texture->getColor(u + 1e-4, v).norm() : h;  // u+ε处高度
    float h_v = payload.texture ? payload.texture->getColor(u, v + 1e-4).norm() : h;  // v+ε处高度
    
    // 计算高度梯度（按注释公式）
    float dU = kh * kn * (h_u - h);
    float dV = kh * kn * (h_v - h);

    // 3. 计算切线空间中的扰动法向量并转换到世界空间
    Eigen::Vector3f ln(-dU, -dV, 1.0f);  // 切线空间法向量
    Eigen::Vector3f n = (TBN * ln).normalized();  // 转换到世界空间并归一化

    // 4. 使用扰动后的法向量进行Blinn-Phong光照计算
    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights)
    {
        // 光源方向向量
        Eigen::Vector3f light_dir = light.position - point;
        float distance = light_dir.norm();
        Eigen::Vector3f L = light_dir.normalized();
        
        // 视线方向向量
        Eigen::Vector3f V = (eye_pos - point).normalized();
        
        // 半程向量
        Eigen::Vector3f H = (L + V).normalized();
        
        // 衰减因子
        float attenuation = 1.0f / (distance * distance);
        
        // 环境光分量
        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);
        
        // 漫反射分量（使用扰动后的法向量）
        float diff_dot = std::max(n.dot(L), 0.0f);
        Eigen::Vector3f diffuse = kd.cwiseProduct(light.intensity) * diff_dot * attenuation;
        
        // 高光分量（使用扰动后的法向量）
        float spec_dot = std::max(n.dot(H), 0.0f);
        Eigen::Vector3f specular = ks.cwiseProduct(light.intensity) * std::pow(spec_dot, p) * attenuation;
        
        // 累加光源贡献
        result_color += ambient + diffuse + specular;
    }

    //Eigen::Vector3f result_color = {0, 0, 0};
    //result_color = normal;

    result_color = result_color.cwiseMin(1.0f).cwiseMax(0.0f);
    return result_color * 255.f;
}

int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "../models/spot/";

    // Load .obj File
    bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj");
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
                t->setVertex(j,Vector4f(mesh.Vertices[i+j].Position.X,mesh.Vertices[i+j].Position.Y,mesh.Vertices[i+j].Position.Z,1.0));
                t->setNormal(j,Vector3f(mesh.Vertices[i+j].Normal.X,mesh.Vertices[i+j].Normal.Y,mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j,Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    auto texture_path = "hmap.jpg";
    r.set_texture(Texture(obj_path + texture_path));

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = phong_fragment_shader;

    if (argc >= 2)
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    Eigen::Vector3f eye_pos = {0,0,10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a' )
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }

    }
    return 0;
}
