#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <string>

// ����PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ���ڳߴ�
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// �����Σ������Σ������?
enum ShapeType {
    TRIANGLE,
    SQUARE,
    PENTAGON
};

// �������Ʊ���
float rotationSpeed = 1.0f;
float rotationAngle = 0.0f;
bool isRotating = true;
ShapeType currentShape = TRIANGLE;
float color[3] = { 1.0f, 0.5f, 0.2f }; // Ĭ����ɫ
float scale = 1.0f;                 // ��������

// �ص���������
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void rotateZ(float angle, float* matrix);
void renderText(GLFWwindow* window, const std::string& text, float x, float y, float scale, float* color);

// ������ɫ��Դ����
const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec3 aColor;\n"
"out vec3 ourColor;\n"
"uniform mat4 transform;\n"
"void main()\n"
"{\n"
"   gl_Position = transform * vec4(aPos, 1.0);\n"
"   ourColor = aColor;\n"
"}\0";

// Ƭ����ɫ��Դ����
const char* fragmentShaderSource = "#version 330 core\n"
"in vec3 ourColor;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   FragColor = vec4(ourColor, 1.0f);\n"
"}\0";

// ��Ⱦ����
void renderText(GLFWwindow* window, const std::string& text, float x, float y, float scale, float* color) {
    // ʹ�ü򵥵��߶�ģ���ı���ʾ
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, SCR_WIDTH, 0, SCR_HEIGHT, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(color[0], color[1], color[2]);
    glRasterPos2f(x, y);

    // �򵥻���һ��������Ϊ�ı�����
    glBegin(GL_QUADS);
    glVertex2f(x - 5, y - 15);
    glVertex2f(x + text.length() * 10 * scale, y - 15);
    glVertex2f(x + text.length() * 10 * scale, y + 5);
    glVertex2f(x - 5, y + 5);
    glEnd();

    // �ָ�����״̬
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

int main() {
    // ��ʼ��GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // ����OpenGL�汾
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // ��������
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "OpenGL Enhanced Interaction", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // ���õ�ǰ������
    glfwMakeContextCurrent(window);

    // ���ô��ڴ�С�仯�ص�
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // ��ʼ��GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // �����ӿ�
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

    // ���붥����ɫ��
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // ��鶥����ɫ������?
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Vertex shader compilation error:\n" << infoLog << std::endl;
        return -1;
    }

    // ����Ƭ����ɫ��
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // ���Ƭ����ɫ������?
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Fragment shader compilation error:\n" << infoLog << std::endl;
        return -1;
    }

    // ������������ɫ������
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // ���������Ӵ���
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader program linking error:\n" << infoLog << std::endl;
        return -1;
    }

    // ɾ����ɫ������
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // ��������
    // ������
    float triangleVertices[] = {
         0.0f,  0.5f, 0.0f,  color[0], color[1], color[2],
         0.5f, -0.5f, 0.0f,  color[0], color[1], color[2],
        -0.5f, -0.5f, 0.0f,  color[0], color[1], color[2]
    };

    // ������
    float squareVertices[] = {
         0.5f,  0.5f, 0.0f,  color[0], color[1], color[2],
         0.5f, -0.5f, 0.0f,  color[0], color[1], color[2],
        -0.5f, -0.5f, 0.0f,  color[0], color[1], color[2],
        -0.5f, -0.5f, 0.0f,  color[0], color[1], color[2],
        -0.5f,  0.5f, 0.0f,  color[0], color[1], color[2],
         0.5f,  0.5f, 0.0f,  color[0], color[1], color[2]
    };

    // �����?
    float pentagonVertices[] = {
         0.0f,  0.5f, 0.0f,  color[0], color[1], color[2],
         0.47f,  0.15f, 0.0f,  color[0], color[1], color[2],
         0.29f, -0.4f, 0.0f,  color[0], color[1], color[2],
        -0.29f, -0.4f, 0.0f,  color[0], color[1], color[2],
        -0.47f,  0.15f, 0.0f,  color[0], color[1], color[2],
         0.0f,  0.5f, 0.0f,  color[0], color[1], color[2]
    };

    // ����VAO��VBO
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    // ��VAO
    glBindVertexArray(VAO);

    // ��VBO����������������
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);

    // ���ö�������ָ�� - λ��
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // ���ö�������ָ�� - ��ɫ
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // ���?
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // ������Ȳ���?
    glEnable(GL_DEPTH_TEST);

    // ��¼��һ֡ʱ��
    float lastFrame = 0.0f;
    int frameCount = 0;
    float fpsTimer = 0.0f;
    float fps = 0.0f;

    // �任����
    float transform[16];

    // ��ѭ��
    while (!glfwWindowShouldClose(window)) {
        // ����֡ʱ���?
        float currentFrame = glfwGetTime();
        float deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // ����FPS
        frameCount++;
        fpsTimer += deltaTime;
        if (fpsTimer >= 1.0f) {
            fps = frameCount / fpsTimer;
            frameCount = 0;
            fpsTimer = 0.0f;
        }

        // ��������
        processInput(window);

        // ����
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ʹ����ɫ������
        glUseProgram(shaderProgram);

        // ������ת�Ƕ�
        if (isRotating) {
            rotationAngle += rotationSpeed * deltaTime * 50.0f; // ����50ʹ��ת������
            if (rotationAngle >= 360.0f) {
                rotationAngle -= 360.0f;
            }
            else if (rotationAngle < 0.0f) {
                rotationAngle += 360.0f;
            }
        }

        // ���¶�������
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        switch (currentShape) {
        case TRIANGLE:
            // ������������ɫ
            for (int i = 0; i < 3; i++) {
                triangleVertices[i * 6 + 3] = color[0];
                triangleVertices[i * 6 + 4] = color[1];
                triangleVertices[i * 6 + 5] = color[2];
            }
            glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);
            break;
        case SQUARE:
            // ������������ɫ
            for (int i = 0; i < 6; i++) {
                squareVertices[i * 6 + 3] = color[0];
                squareVertices[i * 6 + 4] = color[1];
                squareVertices[i * 6 + 5] = color[2];
            }
            glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);
            break;
        case PENTAGON:
            // ������������?
            for (int i = 0; i < 6; i++) {
                pentagonVertices[i * 6 + 3] = color[0];
                pentagonVertices[i * 6 + 4] = color[1];
                pentagonVertices[i * 6 + 5] = color[2];
            }
            glBufferData(GL_ARRAY_BUFFER, sizeof(pentagonVertices), pentagonVertices, GL_STATIC_DRAW);
            break;
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // ��ʼ����λ����
        for (int i = 0; i < 16; i++) {
            transform[i] = 0.0f;
        }
        transform[0] = 1.0f;
        transform[5] = 1.0f;
        transform[10] = 1.0f;
        transform[15] = 1.0f;

        // Ӧ������
        transform[0] *= scale;
        transform[5] *= scale;
        transform[10] *= scale;

        // Ӧ��Z����ת
        rotateZ(rotationAngle, transform);

        // ���任���󴫵ݸ���ɫ��
        unsigned int transformLoc = glGetUniformLocation(shaderProgram, "transform");
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, transform);

        // ���Ƶ�ǰ��״
        glBindVertexArray(VAO);
        switch (currentShape) {
        case TRIANGLE:
            glDrawArrays(GL_TRIANGLES, 0, 3);
            break;
        case SQUARE:
            glDrawArrays(GL_TRIANGLES, 0, 6);
            break;
        case PENTAGON:
            glDrawArrays(GL_TRIANGLE_FAN, 0, 6);
            break;
        }

        // ��Ⱦ��Ϣ���?
        glUseProgram(0); // ������ɫ������ʹ�ù̶����ܹ���
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // ���ư�͸���������?
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(0.2f, 0.2f, 0.2f, 0.8f);
        glBegin(GL_QUADS);
        glVertex2f(10, SCR_HEIGHT - 10);
        glVertex2f(300, SCR_HEIGHT - 10);
        glVertex2f(300, SCR_HEIGHT - 150);
        glVertex2f(10, SCR_HEIGHT - 150);
        glEnd();

        // ��ʾ������Ϣ
        float textColor[3] = { 1.0f, 1.0f, 1.0f };
        std::string shapeStr;
        switch (currentShape) {
        case TRIANGLE: shapeStr = "Triangle"; break;
        case SQUARE: shapeStr = "Square"; break;
        case PENTAGON: shapeStr = "Pentagon"; break;
        }

        renderText(window, "Shape: " + shapeStr, 20, SCR_HEIGHT - 30, 1.0f, textColor);
        renderText(window, "Rotation Speed: " + std::to_string((int)(rotationSpeed * 10) / 10.0f), 20, SCR_HEIGHT - 55, 1.0f, textColor);
        renderText(window, "Rotation: " + std::string(isRotating ? "On" : "Off"), 20, SCR_HEIGHT - 80, 1.0f, textColor);
        renderText(window, "Scale: " + std::to_string((int)(scale * 10) / 10.0f), 20, SCR_HEIGHT - 105, 1.0f, textColor);
        renderText(window, "FPS: " + std::to_string((int)fps), 20, SCR_HEIGHT - 130, 1.0f, textColor);

        // ��ʾ������ʾ
        renderText(window, "Press 1-3 to change shape", 20, 30, 0.8f, textColor);
        renderText(window, "W/S to change scale", 20, 55, 0.8f, textColor);
        renderText(window, "R/G/B to change color", 20, 80, 0.8f, textColor);
        renderText(window, "Arrow keys to control rotation", 20, 105, 0.8f, textColor);
        renderText(window, "Space to pause rotation", 20, 130, 0.8f, textColor);
        renderText(window, "ESC to exit", 20, 155, 0.8f, textColor);

        glDisable(GL_BLEND);

        // �����������������¼�
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // ������Դ
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}

// ���ڴ�С�仯�ص�
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// ���봦������
void processInput(GLFWwindow* window) {
    // ��ESC���رմ���
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // ���ո�����?������ת
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        // ���Ӽ򵥵İ�������
        static double lastPress = 0;
        double currentTime = glfwGetTime();
        if (currentTime - lastPress > 0.3) {
            isRotating = !isRotating;
            lastPress = currentTime;
        }
    }

    // ���ϼ�ͷ������ת�ٶ�
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        rotationSpeed += 0.1f * glfwGetTime() * 0.01f;

    // ���¼�ͷ��С��ת�ٶ�
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS && rotationSpeed > 0.1f)
        rotationSpeed -= 0.1f * glfwGetTime() * 0.01f;

    // �����ͷ��ת��ת����?
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        static double lastPress = 0;
        double currentTime = glfwGetTime();
        if (currentTime - lastPress > 0.3) {
            rotationSpeed = -fabs(rotationSpeed);
            lastPress = currentTime;
        }
    }

    // ���Ҽ�ͷ�ָ�˳ʱ����ת
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        static double lastPress = 0;
        double currentTime = glfwGetTime();
        if (currentTime - lastPress > 0.3) {
            rotationSpeed = fabs(rotationSpeed);
            lastPress = currentTime;
        }
    }

    // ���ּ�1-3�л���״
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
        static double lastPress = 0;
        double currentTime = glfwGetTime();
        if (currentTime - lastPress > 0.3) {
            currentShape = TRIANGLE;
            lastPress = currentTime;
        }
    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
        static double lastPress = 0;
        double currentTime = glfwGetTime();
        if (currentTime - lastPress > 0.3) {
            currentShape = SQUARE;
            lastPress = currentTime;
        }
    }
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
        static double lastPress = 0;
        double currentTime = glfwGetTime();
        if (currentTime - lastPress > 0.3) {
            currentShape = PENTAGON;
            lastPress = currentTime;
        }
    }

    // W/S����������
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS && scale < 2.0f)
        scale += 0.005f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS && scale > 0.3f)
        scale -= 0.005f;

    // R/G/B��������ɫ
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        color[0] += 0.005f;
        if (color[0] > 1.0f) color[0] = 1.0f;
    }
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
        color[1] += 0.005f;
        if (color[1] > 1.0f) color[1] = 1.0f;
    }
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) {
        color[2] += 0.005f;
        if (color[2] > 1.0f) color[2] = 1.0f;
    }
    // Shift+R/G/B��С��ɫ����
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            color[0] -= 0.005f;
            if (color[0] < 0.0f) color[0] = 0.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
            color[1] -= 0.005f;
            if (color[1] < 0.0f) color[1] = 0.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) {
            color[2] -= 0.005f;
            if (color[2] < 0.0f) color[2] = 0.0f;
        }
    }
}

// ʵ��Z����ת����
void rotateZ(float angle, float* matrix) {
    // ���Ƕ�ת��Ϊ����
    float rad = angle * M_PI / 180.0f;
    float cosA = cos(rad);
    float sinA = sin(rad);

    // ���浱ǰ�����ǰ���У�������ת�����޸ģ�?
    float m0 = matrix[0], m4 = matrix[4], m8 = matrix[8], m12 = matrix[12];
    float m1 = matrix[1], m5 = matrix[5], m9 = matrix[9], m13 = matrix[13];

    // Ӧ����ת����
    matrix[0] = m0 * cosA - m1 * sinA;
    matrix[4] = m4 * cosA - m5 * sinA;
    matrix[8] = m8 * cosA - m9 * sinA;
    matrix[12] = m12 * cosA - m13 * sinA;

    matrix[1] = m0 * sinA + m1 * cosA;
    matrix[5] = m4 * sinA + m5 * cosA;
    matrix[9] = m8 * sinA + m9 * cosA;
    matrix[13] = m12 * sinA + m13 * cosA;
}
