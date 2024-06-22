#version 430 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D depthMap;

void main()
{
    float depth = texture(depthMap, TexCoords).r;
    FragColor = vec4(vec3(depth), 1.0);
}
