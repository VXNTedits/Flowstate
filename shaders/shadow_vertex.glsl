#version 430 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoords;

uniform mat4 model;
uniform mat4 lightSpaceMatrix;
uniform mat4 view;
uniform mat4 projection;

out vec4 FragPosLightSpace;
out vec2 TexCoords;

void main()
{
    vec4 worldPosition = model * vec4(aPos, 1.0);
    FragPosLightSpace = lightSpaceMatrix * worldPosition;
    gl_Position = projection * view * worldPosition;
    TexCoords = aTexCoords;
}
