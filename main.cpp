#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
//#include <glad/glad.h>
//#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>


#include "stb/stb_image.h"
#include "tools.h"
#include "LocalWarp.h"
#include "GlobalWarp.h"

#define clamp(x,a,b)    (  ((a)<(b))				\
? ((x)<(a))?(a):(((x)>(b))?(b):(x))	\
: ((x)<(b))?(b):(((x)>(a))?(a):(x))	\
)

using namespace std;

//void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
//    glViewport(0, 0, width, height);
//}
//
//void processInput(GLFWwindow* window) {
//    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
//        glfwSetWindowShouldClose(window, true);
//    }
//}

GLuint texGround;
vector<pair<int, int>> mesh;
vector<pair<int, int>> final_mesh;
int vertexRow;
int vertexCol;
Mat global_img;

GLuint matToTexture(cv::Mat mat, GLenum minFilter = GL_LINEAR,
                    GLenum magFilter = GL_LINEAR, GLenum wrapFilter = GL_REPEAT) {
    //cv::flip(mat, mat, 0);
    // Generate a number for our textureID's unique handle
    GLuint textureID;
    glGenTextures(1, &textureID);

    // Bind to our texture handle
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Catch silly-mistake texture interpolation method for magnification
    if (magFilter == GL_LINEAR_MIPMAP_LINEAR ||
        magFilter == GL_LINEAR_MIPMAP_NEAREST ||
        magFilter == GL_NEAREST_MIPMAP_LINEAR ||
        magFilter == GL_NEAREST_MIPMAP_NEAREST)
    {
        //cout << "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR" << endl;
        magFilter = GL_LINEAR;
    }

    // Set texture interpolation methods for minification and magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

    // Set texture clamping method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

    // Set incoming texture format to:
    // GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
    // GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
    // Work out other mappings as required ( there's a list in comments in main() )
    GLenum inputColourFormat = GL_BGR_EXT;
    if (mat.channels() == 1)
    {
        inputColourFormat = GL_LUMINANCE;
    }

    // Create the texture
    glTexImage2D(GL_TEXTURE_2D,     // Type of texture
                 0,                 // Pyramid level (for mip-mapping) - 0 is the top level
                 GL_RGB,            // Internal colour format to convert to
                 mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
                 mat.rows,          // Image height i.e. 480 for Kinect in standard mode
                 0,                 // Border width in pixels (can either be 1 or 0)
                 inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
                 GL_UNSIGNED_BYTE,  // Image data type
                 mat.ptr());        // The actual image data itself

    return textureID;
}

void display() {
    glLoadIdentity();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texGround);
    vector<PDD> d_mesh;
    vector<PDD> d_final_mesh;
    for (int row = 0; row < vertexRow; row ++) {
        for (int col = 0; col < vertexCol; col ++) {
            int index = row * vertexCol + col;
            PDD d_coord = {final_mesh[index].first, final_mesh[index].second};
            PDD d_localcoord = {mesh[index].first, mesh[index].second};
            d_coord.first /= global_img.rows;
            d_coord.second /= global_img.cols;
            d_coord.first -= 0.5;
            d_coord.second -= 0.5;
            d_coord.first *= 2;
            d_coord.second *= 2;
            d_coord =  {clamp(d_coord.first, -1, 1), clamp(d_coord.second, -1, 1)};
            //cout << coord << " ";

            d_localcoord.first /= global_img.rows;
            d_localcoord.second /= global_img.cols;
            d_localcoord = {clamp(d_localcoord.first, 0, 1), clamp(d_localcoord.second, 0, 1)};

            d_final_mesh.push_back(d_coord);
            d_mesh.push_back(d_localcoord);
        }
    }

    for (int row = 0; row < vertexRow - 1; row ++) {
        for (int col = 0; col < vertexCol - 1; col ++) {
            int index = row * vertexCol + col;
            PDD local_left_top = d_mesh[index];
            PDD local_right_top = d_mesh[index + 1];
            PDD local_left_bottom = d_mesh[index + vertexCol];
            PDD local_right_bottom = d_mesh[index + vertexCol + 1];


            PDD global_left_top = d_final_mesh[index];
            PDD global_right_top = d_final_mesh[index + 1];
            PDD global_left_bottom = d_final_mesh[index + vertexCol];
            PDD global_right_bottom = d_final_mesh[index + vertexCol + 1];


            glBegin(GL_QUADS);
            glTexCoord2d(local_right_top.second, local_right_top.first); glVertex3d(global_right_top.second,  -1 * global_right_top.first, 0.0f);
            glTexCoord2d(local_right_bottom.second, local_right_bottom.first); glVertex3d(global_right_bottom.second,  -1 * global_right_bottom.first, 0.0f);
            glTexCoord2d(local_left_bottom.second, local_left_bottom.first);	glVertex3d(global_left_bottom.second,  -1 * global_left_bottom.first, 0.0f);
            glTexCoord2d(local_left_top.second, local_left_top.first); glVertex3d(global_left_top.second,  -1 * global_left_top.first, 0.0f);
            glEnd();

        }
    }
    glutSwapBuffers();
}


int main(int argc, char* argv[]) {

    Tools tools;
    Mat img = imread("../pic/2_input.jpg");

    Mat input_img;
    double scale = tools.shrinkImage(img, input_img);

    int meshRow = tools.get_mesh_size(input_img).first;
    int meshCol = tools.get_mesh_size(input_img).second;
    Mat mask = tools.createMask(input_img);

    // local warp
    cout << "start local warp" << "\t";
    LocalWarp localWarp(input_img, mask);
    vector<pair<int, int>> vertexes = localWarp.get_warp_mesh(meshRow, meshCol);
    cout << "finish" << endl;

    // global warp
    GlobalWarp globalWarp(vertexes, input_img, meshRow, meshCol, 100, 1e8);
    auto final_vertexes = globalWarp.get_vertexes();

    //放大mesh到原图
    for (auto &vertex: final_vertexes) {
        int x = vertex.first;
        int y = vertex.second;
        vertex.first = lround(x * scale);
        vertex.second = lround(y * scale);
    }
    for (auto &vertex: vertexes) {
        int x = vertex.first;
        int y = vertex.second;
        vertex.first = lround(x * scale);
        vertex.second = lround(y * scale);
    }

    mesh = vertexes;
    final_mesh = final_vertexes;
    vertexRow = meshRow + 1;
    vertexCol = meshCol + 1;
    global_img = img;

    // 展示
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(img.cols, img.rows);
    glutCreateWindow("img");
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // 防止图片倾斜
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    texGround = matToTexture(img);
    glutDisplayFunc(&display);
    glutMainLoop();


    return 0;
}