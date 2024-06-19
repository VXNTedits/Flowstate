#version 330 core

in vec4 FragPosLightSpace;
in vec2 TexCoords;

uniform sampler2D shadowMap;
uniform vec3 lightColor;
uniform vec3 objectColor;

out vec4 FragColor;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;

    float bias = 0.005;
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    return shadow;
}


void main()
{
    float shadow = ShadowCalculation(FragPosLightSpace);
    vec3 color = (1.0 - shadow) * lightColor * objectColor;
    FragColor = vec4(color, 1.0);
}
