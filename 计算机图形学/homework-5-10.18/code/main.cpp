#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>  // 用于距离计算

std::vector<cv::Point2f> control_points;

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < 4) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }     
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t) 
{
    if (control_points.size() == 1)
    {
        return control_points[0];
    }

    std::vector<cv::Point2f> next_level;
    for (size_t i = 0; i < control_points.size() - 1; ++i)
    {
        cv::Point2f new_point = control_points[i] * (1 - t) + control_points[i + 1] * t;
        next_level.push_back(new_point);
    }

    return recursive_bezier(next_level, t);
}

// 反走样绘制：根据距离对周围像素进行加权着色
void draw_antialiased_point(cv::Mat &window, float x, float y, float intensity) 
{
    // 获取当前点所在的像素坐标（整数部分）
    int px = static_cast<int>(std::floor(x));
    int py = static_cast<int>(std::floor(y));

    // 计算点到像素中心的偏移量（[0,1)范围）
    float dx = x - (px + 0.5f);
    float dy = y - (py + 0.5f);

    // 遍历周围4个像素（当前像素及右、下、右下相邻像素）
    for (int i = 0; i < 2; ++i) 
    {
        for (int j = 0; j < 2; ++j) 
        {
            int cx = px + i;
            int cy = py + j;

            // 检查像素是否在窗口范围内
            if (cx < 0 || cx >= window.cols || cy < 0 || cy >= window.rows)
                continue;

            // 计算当前子像素中心到曲线点的距离
            float dist = std::sqrt(std::pow(dx - i + 0.5f, 2) + std::pow(dy - j + 0.5f, 2));
            
            // 距离越近，权重越大（距离超过0.5√2时权重为0）
            float weight = std::max(0.0f, 1.0f - dist * std::sqrt(2.0f));
            
            // 累积绿色分量（乘以intensity控制基础亮度）
            int green = static_cast<int>(window.at<cv::Vec3b>(cy, cx)[1] + weight * intensity);
            window.at<cv::Vec3b>(cy, cx)[1] = static_cast<uchar>(std::min(255, green));
        }
    }
}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    // 减小步长以获得更密集的采样点，提升反走样效果
    for (double t = 0.0; t <= 1.0; t += 0.0005)
    {
        cv::Point2f curve_point = recursive_bezier(control_points, static_cast<float>(t));
        // 调用反走样绘制函数（基础亮度设为128，避免叠加过度饱和）
        draw_antialiased_point(window, curve_point.x, curve_point.y, 128.0f);
    }
}

int main() 
{
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) 
    {
        for (auto &point : control_points) 
        {
            cv::circle(window, point, 3, cv::Scalar(255, 255, 255), 3);
        }

        if (control_points.size() == 4) 
        {
            // 如需对比，可取消下行注释（红色曲线无反走样）
            // naive_bezier(control_points, window);
            bezier(control_points, window);  // 带反走样的绿色曲线

            cv::imshow("Bezier Curve", window);
            cv::imwrite("antialiased_bezier.png", window);
            key = cv::waitKey(0);
            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

    return 0;
}