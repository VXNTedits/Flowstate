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

void main()
{
    vec3 ambient = 0.1 * objectColor;

    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 result = ambient;

    for (int i = 0; i < NUM_LIGHTS; i++) {
        vec3 lightDir = normalize(lights[i].position - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lights[i].color * objectColor;

        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        vec3 specular = spec * lights[i].color * specularColor;

        result += diffuse + specular;
    }

    FragColor = vec4(result, 1.0);
}
