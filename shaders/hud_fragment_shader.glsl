#version 430 core
in vec2 TexCoord;

out vec4 color;

uniform sampler2D overlayTexture;

void main()
{
    color = texture(overlayTexture, TexCoord);
}
