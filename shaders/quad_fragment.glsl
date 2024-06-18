#version 330 core
in vec2 TexCoords;
out vec4 FragColor;

uniform vec3 quadColor;

void main()
{
    FragColor = vec4(quadColor, 1.0);
}
