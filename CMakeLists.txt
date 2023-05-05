cmake_minimum_required(VERSION 3.23)
project(recwarp)

set(CMAKE_CXX_STANDARD 17)

# glad
include_directories(./glad/include)

set(LOCAL_H /opt/homebrew/include)
include_directories(${LOCAL_H})

set(GLEW_H /opt/homebrew/Cellar/glew/2.2.0_1/include/GL)
set(GLFW_H /opt/homebrew/Cellar/glfw/3.3.8/include/GLFW)
link_directories(${GLEW_H} ${GLFW_H})

set(GLEW_LINK /opt/homebrew/Cellar/glew/2.2.0_1/lib/libGLEW.2.2.dylib)
set(GLFW_LINK /opt/homebrew/Cellar/glfw/3.3.8/lib/libglfw.3.dylib)
link_libraries(${OPENGL} ${GLEW_LINK} ${GLFW_LINK})

add_executable(recwarp main.cpp ./glad/src/glad.c stb/stb_image.h stb/stb_image.cpp Shader.h)

if (APPLE)
    target_link_libraries(recwarp "-framework OpenGL")
    target_link_libraries(recwarp "-framework GLUT")
endif ()