#version 430 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D depthMap;

void main()
{
    float depthValue = texture(depthMap, TexCoords).r;
    // Visualize depth value (0.0 to 1.0) as a grayscale color
    FragColor = vec4(vec3(depthValue), 1.0);
}
