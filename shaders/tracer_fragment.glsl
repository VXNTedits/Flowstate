//#version 430 core
//out vec4 FragColor;
//
//uniform vec3 lineColor;
//
//void main() {
//    FragColor = vec4(lineColor, 1.0);
//}
#version 430 core
out vec4 FragColor;

uniform vec3 lightPos; // Position of the light source
uniform vec3 viewPos;  // Position of the camera/viewer
uniform vec3 lightColor; // Color of the light
uniform float lightIntensity; // Intensity of the light
uniform vec3 tracerColor;

in vec3 FragPos; // Position of the fragment

void main()
{
    // Ambient lighting
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse lighting
    vec3 norm = normalize(FragPos); // Assuming normal is in the direction of the fragment position
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular lighting
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    // Combine all the lighting components
    vec3 result = (ambient + diffuse + specular) * tracerColor * lightIntensity;
    FragColor = vec4(result, 1.0);
}
