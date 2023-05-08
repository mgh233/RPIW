#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>


#include "stb/stb_image.h"
#include "myShader.h"
#include "tools.h"
#include "LocalWarp.h"

using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

int main() {

//    // 初始化窗口
//    glfwInit();
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
//    GLFWwindow* window = glfwCreateWindow(800, 600, "window", NULL, NULL);
//    if (window == NULL) {
//        cout << "Failed to create window" << endl;
//        glfwTerminate();
//        return -1;
//    }
//    glfwMakeContextCurrent(window);
//
//    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
//        cout << "Failed to initialize GLAD" << endl;
//        return -1;
//    }
//
//    // 绑定回调函数
//    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
//
//    // shader
//    Shader ourShader("../texture.vs", "../texture.fs");
//
//    // 防止图片出现变形和颠倒
//    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//    stbi_set_flip_vertically_on_load(true);
//
//    // 生成矩形
//    float vertices[] = {
////     ---- 位置 ----       ---- 颜色 ----     - 纹理坐标 -
//            0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // 右上
//            0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,   // 右下
//            -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,   // 左下
//            -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f    // 左上
//    };
//    unsigned int indices[] = {
//            0, 1, 3, // first triangle
//            1, 2, 3  // second triangle
//    };
//    unsigned int VBO, VAO, EBO;
//    glGenVertexArrays(1, &VAO);
//    glGenBuffers(1, &VBO);
//    glGenBuffers(1, &EBO);
//
//    glBindVertexArray(VAO);
//
//    glBindBuffer(GL_ARRAY_BUFFER, VBO);
//    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
//
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
//    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
//
//    // position attribute
//    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
//    glEnableVertexAttribArray(0);
//    // color attribute
//    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
//    glEnableVertexAttribArray(1);
//    // texture coord attribute
//    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
//    glEnableVertexAttribArray(2);
//
//    // 加载图片并生成纹理
//    int width, height, nrChannels;
//    unsigned char *data = stbi_load("../pic/2_input.jpg", &width, &height, &nrChannels, 0);
//
//    unsigned int texture;
//    glGenTextures(1, &texture);
//    glBindTexture(GL_TEXTURE_2D, texture);
//    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
//    glGenerateMipmap(GL_TEXTURE_2D);
//    stbi_image_free(data);
//
//    // 循环
//    while (!glfwWindowShouldClose(window)) {
//        // 输入
//        processInput(window);
//
//        // 渲染指令
//        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
//        glClear(GL_COLOR_BUFFER_BIT);
//
//        glBindTexture(GL_TEXTURE_2D, texture);
//
//        ourShader.use();
//        glBindVertexArray(VAO);
//        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
//
//        // 检查并调用事件，交换缓冲
//        glfwSwapBuffers(window);
//        glfwPollEvents();
//    }
//
//    glfwTerminate();

    Mat img = imread("../pic/2_input.jpg");
    Tools tools;
    Mat mask = tools.createMask(img);
    LocalWarp localWarp(img, mask);
    localWarp.get_warp_mesh(20, 20);
    return 0;
}