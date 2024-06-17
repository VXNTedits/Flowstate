#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

struct Light {
    vec3 position;
    vec3 color;
};

#define NUM_LIGHTS 3

uniform Light lights[NUM_LIGHTS];
uniform vec3 viewPos;
uniform vec3 objectColor;
uniform vec3 specularColor;
uniform float shininess;

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

void main()
{
    vec3 ambient = 0.1 * objectColor;

    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 result = ambient;

    vec3 color = objectColor * noise(FragPos * 0.1);

    for (int i = 0; i < NUM_LIGHTS; i++) {
        vec3 lightDir = normalize(lights[i].position - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lights[i].color * color;

        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        vec3 specular = spec * lights[i].color * specularColor;

        result += diffuse + specular;
    }

    FragColor = vec4(result, 1.0);
}
