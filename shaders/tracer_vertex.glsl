//#version 330 core
//layout(location = 0) in vec3 aPos;  // Position of vertices
//
//uniform mat4 model;
//uniform mat4 view;
//uniform mat4 projection;
//
//void main() {
//    gl_Position = projection * view * model * vec4(aPos, 1.0);
//}
#version 330 core
layout(location = 0) in vec3 aPos;  // Position of vertices

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
