//#version 330 core
//out vec4 FragColor;
//
//uniform vec3 tracerColor;  // Color of the tracer
//
//void main() {
//    // Calculate the distance from the center of the fragment in NDC space
//    vec2 ndc = gl_FragCoord.xy / vec2(800.0, 600.0);  // Assuming a viewport of 800x600, adjust if different
//    float distanceFromCenter = length(ndc - vec2(0.5, 0.5));
//
//    // Simulated glow factor
//    float alpha = 1.0 - distanceFromCenter * 2.0;  // Adjust multiplier for desired effect
//    alpha = clamp(alpha, 0.0, 1.0);  // Ensure alpha stays within valid bounds
//
//    FragColor = vec4(tracerColor * alpha, alpha);
//}
#version 330 core
out vec4 FragColor;

in vec3 FragPos;  // Fragment position in world space
in vec3 Normal;   // Normal at the fragment

uniform vec3 viewPos;  // Camera position
uniform vec3 tracerColor;  // Color of the tracer
uniform int numTracers;  // Number of tracers

struct Light {
    vec3 position;
    vec3 color;
    float intensity;
};

uniform Light tracers[100];  // Array of tracer lights, adjust the size as needed

void main() {
    // Ambient lighting
    vec3 ambient = 0.1 * tracerColor;

    // Initialize lighting
    vec3 lighting = ambient;

    // Loop through each tracer light
    for (int i = 0; i < numTracers; i++) {
        // Calculate the direction and distance to the light
        vec3 lightDir = normalize(tracers[i].position - FragPos);
        float distance = length(tracers[i].position - FragPos);

        // Diffuse lighting
        float diff = max(dot(Normal, lightDir), 0.0);
        vec3 diffuse = diff * tracers[i].color * tracers[i].intensity / (distance * distance);

        // Specular lighting
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, Normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = spec * tracers[i].color * tracers[i].intensity / (distance * distance);

        // Accumulate lighting
        lighting += diffuse + specular;
    }

    // Output the final color
    FragColor = vec4(lighting, 1.0);
}
