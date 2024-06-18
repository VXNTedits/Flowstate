#version 330 core

layout(location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 lightSpaceMatrix;
in vec2 TexCoords;
uniform sampler2D shadowMap;
out vec4 FragColor;
//uniform mat4 view;
//uniform mat4 projection;

void main()
{
    vec4 shadowColor = texture(shadowMap, TexCoords);
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
    FragColor = shadowColor;
}
