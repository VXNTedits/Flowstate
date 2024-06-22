#version 430 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D sceneTexture;
uniform sampler2D volumetricTexture;

void main()
{
    vec4 sceneColor = texture(sceneTexture, TexCoords);
    vec4 volumetricColor = texture(volumetricTexture, TexCoords);
    FragColor = sceneColor + volumetricColor * volumetricColor.a;
}
