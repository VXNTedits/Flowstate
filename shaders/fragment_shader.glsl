#version 430 core

#define NUM_LIGHTS 3

out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec4 FragPosLightSpace[NUM_LIGHTS];

struct Light {
    vec3 position;
    vec3 color;
};

uniform Light lights[NUM_LIGHTS];
uniform vec3 viewPos;
uniform vec3 objectColor;
uniform vec3 specularColor;
uniform float shininess;
uniform float roughness;
uniform float bumpScale;
uniform sampler2D shadowMap;
uniform bool useTexture;
uniform sampler2D texture1;
uniform bool enableBumpMapping;  // New uniform to control bump mapping

float hash(float n) { return fract(sin(n) * 43758.5453); }

float noise(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+ 0.0), hash(n+ 1.0),f.x),
                   mix( hash(n+57.0), hash(n+58.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}

float getHeight(vec3 pos) {
    return noise(pos * bumpScale);
}

vec3 getPerturbedNormal(vec3 pos, vec3 norm) {
    float hL = getHeight(pos + vec3(-0.01, 0.0, 0.0));
    float hR = getHeight(pos + vec3(0.01, 0.0, 0.0));
    float hD = getHeight(pos + vec3(0.0, -0.01, 0.0));
    float hU = getHeight(pos + vec3(0.0, 0.01, 0.0));
    vec3 dx = vec3(0.02, hR - hL, 0.0);
    vec3 dy = vec3(0.0, hU - hD, 0.02);
    vec3 bumpNormal = normalize(cross(dx, dy));
    return normalize(norm + bumpNormal);
}

float shadowCalculation(vec4 fragPosLightSpace, vec3 norm, vec3 lightDir)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float currentDepth = projCoords.z;
    float bias = max(0.005 * (1.0 - dot(norm, lightDir)), 0.005);

    // PCF (Percentage Closer Filtering)
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
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
    vec3 ambient = 0.01 * objectColor;

    vec3 norm = normalize(Normal);
    vec3 perturbedNormal = norm;

    if (enableBumpMapping) {
        perturbedNormal = getPerturbedNormal(FragPos, norm);
    }

    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 result = ambient;

    vec3 color = objectColor;
    if (useTexture) {
        color *= texture(texture1, FragPos.xy).rgb;
    } else {
        color *= noise(FragPos * 0.05);
    }

    for (int i = 0; i < NUM_LIGHTS; i++) {
        vec3 lightDir = normalize(lights[i].position - FragPos);
        float diff = max(dot(perturbedNormal, lightDir), 0.0);
        vec3 diffuse = diff * lights[i].color * color;

        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(perturbedNormal, halfwayDir), 0.0), shininess * (1.0 - roughness));
        vec3 specular = spec * lights[i].color * specularColor;

        float shadow = shadowCalculation(FragPosLightSpace[i], perturbedNormal, lightDir);
        result += (diffuse + specular) * (1.0 - shadow);
    }

    FragColor = vec4(result, 1.0);
}
