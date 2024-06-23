#version 430 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D sceneTexture;
uniform sampler2D volumetricTexture;
uniform sampler2D tracers;

void main()
{
    vec4 sceneColor = texture(sceneTexture, TexCoords);
    vec4 volumetricColor = texture(volumetricTexture, TexCoords);
    vec4 tracerColor = texture(tracers, TexCoords);
    FragColor = sceneColor + volumetricColor * volumetricColor.a + tracerColor;
}
