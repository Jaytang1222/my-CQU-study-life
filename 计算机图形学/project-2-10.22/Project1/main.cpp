#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

// 窗口参数
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// 核心参数
vector<pair<float, float>> controlPoints;
vector<float> knots;
int degree = 3;        
int sampleCount = 100;   

// 交互状态
bool isDragging = false;
int selectedPoint = -1;
float translateX = 0, translateY = 0;
float scale = 1.0f;
float rotateAngle = 0.0f;

// 函数声明
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void render();
pair<float, float> deBoor(int k, int i, float t);
vector<pair<float, float>> computeBSpline();
void generateKnots();
void initControlPoints();

// 生成均匀节点向量
void generateKnots() {
    knots.clear();
    int n = controlPoints.size();
    if (n < degree) return;

    int m = n + degree;  
    for (int i = 0; i <= m; ++i) {
        if (i < degree) knots.push_back(0.0f);
        else if (i > m - degree) knots.push_back(1.0f);
        else knots.push_back((float)(i - degree + 1) / (n - degree + 2));
    }
}

// 初始化控制点
void initControlPoints() {
    controlPoints = {
        {200.0f, 300.0f},
        {300.0f, 400.0f},
        {400.0f, 300.0f},
        {500.0f, 400.0f}  
    };
    generateKnots();
}

// de-Boor算法
pair<float, float> deBoor(int k, int i, float t) {
    if (k == 1) return controlPoints[i];

    float u_i = knots[i];
    float u_ik = knots[i + k - 1];
    if (fabs(u_ik - u_i) < 1e-6) return controlPoints[i]; 
    float alpha = (t - u_i) / (u_ik - u_i);

    auto p1 = deBoor(k - 1, i, t);
    auto p2 = deBoor(k - 1, i + 1, t);

    return {
        (1 - alpha) * p1.first + alpha * p2.first,
        (1 - alpha) * p1.second + alpha * p2.second
    };
}

// 计算曲线采样点
vector<pair<float, float>> computeBSpline() {
    vector<pair<float, float>> samples;
    int n = controlPoints.size();
    if (n < degree || knots.empty()) return samples;

    float tStart = knots[degree - 1];
    float tEnd = knots[n];
    float step = (tEnd - tStart) / sampleCount;

    for (int s = 0; s <= sampleCount; ++s) {
        float t = tStart + s * step;
        int i;
        for (i = degree - 1; i < n; ++i) {
            if (t <= knots[i + 1]) break;
        }
        samples.push_back(deBoor(degree, i - degree + 1, t));
    }
    return samples;
}

// 渲染函数
void render() {
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translateX, translateY, 0.0f);
    glScalef(scale, scale, 1.0f);
    glRotatef(rotateAngle, 0.0f, 0.0f, 1.0f);

    // 绘制控制点
    glColor3f(1.0f, 0.0f, 0.0f);
    glPointSize(8.0f);
    glBegin(GL_POINTS);
    for (const auto& p : controlPoints) glVertex2f(p.first, p.second);
    glEnd();

    // 绘制控制多边形
    if (controlPoints.size() > 1) {
        glColor3f(0.5f, 0.5f, 0.5f);
        glBegin(GL_LINE_STRIP);
        for (const auto& p : controlPoints) glVertex2f(p.first, p.second);
        glEnd();
    }

    // 绘制B样条曲线
    auto samples = computeBSpline();
    if (samples.size() > 1) {
        glColor3f(0.0f, 0.0f, 1.0f);
        glLineWidth(2.0f);
        glBegin(GL_LINE_STRIP);
        for (const auto& s : samples) glVertex2f(s.first, s.second);
        glEnd();
    }
}

// 窗口尺寸回调
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, 0, height, -1, 1); 
}

// 鼠标点击回调
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        y = SCR_HEIGHT - y;  

        float invScale = 1.0f / scale;
        for (int i = 0; i < controlPoints.size(); ++i) {
            float screenX = controlPoints[i].first * scale + translateX;
            float screenY = controlPoints[i].second * scale + translateY;
            float dx = x - screenX, dy = y - screenY;
            if (sqrt(dx * dx + dy * dy) < 10) {  
                isDragging = true;
                selectedPoint = i;
                break;
            }
        }
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        isDragging = false;
    }
}

// 鼠标移动回调
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (isDragging && selectedPoint != -1) {
        float y = SCR_HEIGHT - ypos;
        float invScale = 1.0f / scale;
        controlPoints[selectedPoint] = {
            (xpos - translateX) * invScale,
            (y - translateY) * invScale
        };
        generateKnots();
    }
}

// 滚轮缩放回调
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    scale *= (yoffset > 0) ? 1.1f : 0.9f;
}

// 键盘回调
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) {
        case GLFW_KEY_W: translateY += 10; break;
        case GLFW_KEY_S: translateY -= 10; break;
        case GLFW_KEY_A: translateX -= 10; break;
        case GLFW_KEY_D: translateX += 10; break;
        case GLFW_KEY_R: rotateAngle += 5; break;
        case GLFW_KEY_F: rotateAngle -= 5; break;
        case GLFW_KEY_N: {  
            double x, y;
            glfwGetCursorPos(window, &x, &y);
            float ry = SCR_HEIGHT - y;
            float invScale = 1.0f / scale;
            controlPoints.push_back({
                (x - translateX) * invScale,
                (ry - translateY) * invScale
                });
            generateKnots();
            break;
        }
        case GLFW_KEY_C:  
            controlPoints.clear();
            controlPoints = { {200,300}, {300,400}, {400,300} };
            generateKnots();
            break;
        }
    }
}

int main() {
    // 初始化GLFW
    if (!glfwInit()) {
        cerr << "GLFW初始化失败" << endl;
        return -1;
    }

    // 兼容模式配置
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE); 

    // 创建窗口
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "B样条曲线", NULL, NULL);
    if (!window) {
        cerr << "窗口创建失败" << endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // 初始化GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        cerr << "GLEW初始化失败" << endl;
        return -1;
    }

    // 设置回调和初始参数
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);
    framebuffer_size_callback(window, SCR_WIDTH, SCR_HEIGHT);

    // 初始化控制点
    initControlPoints();

    // 渲染循环
    while (!glfwWindowShouldClose(window)) {
        render();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}