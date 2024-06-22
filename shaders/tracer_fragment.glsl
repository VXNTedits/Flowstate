#version 430 core

in flat int vertexID;
out vec4 FragColor;

void main() {
    // Debugging vertexID by setting a color based on it
    FragColor = vec4(float(vertexID) / 3.0, 0.0, 1.0 - float(vertexID) / 3.0, 1.0);
}



//#version 430 core
//out vec4 FragColor;
//
//in vec3 FragPos;  // Fragment position in world space
//in vec3 Normal;   // Normal at the fragment
//
//uniform vec3 viewPos;  // Camera position
//uniform vec3 tracerColor;  // Color of the tracer
//uniform int numTracers;  // Number of tracers
//
//layout(std430, binding = 0) buffer TrajectoryBuffer {
//    vec3 trajectories[];
//};
//
//void main() {
//    // Ambient lighting
//    vec3 ambient = 0.3 * tracerColor; // Increased ambient light
//
//    // Initialize lighting
//    vec3 lighting = ambient;
//
//    // Loop through each tracer
//    int trajectoryCount = 256; // Assuming a fixed maximum number of trajectories per tracer
//    for (int i = 0; i < numTracers; i++) {
//        // Loop through each trajectory within the tracer
//        for (int j = 0; j < trajectoryCount; j++) {
//            vec3 trajectory = trajectories[i * trajectoryCount + j];
//            // Check if the trajectory is valid (e.g., not a zero vector)
//            if (trajectory == vec3(0.0, 0.0, 0.0)) continue;
//
//            // Calculate the direction and distance to the light
//            vec3 lightDir = normalize(trajectory - FragPos);
//            float distance = length(trajectory - FragPos);
//
//            // Diffuse lighting
//            float diff = max(dot(Normal, lightDir), 0.0);
//            vec3 diffuse = diff * tracerColor / (distance * distance);
//
//            // Specular lighting
//            vec3 viewDir = normalize(viewPos - FragPos);
//            vec3 reflectDir = reflect(-lightDir, Normal);
//            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
//            vec3 specular = spec * tracerColor / (distance * distance);
//
//            // Accumulate lighting
//            lighting += diffuse + specular;
//        }
//    }
//
//    // Output the final color
//    FragColor = vec4(lighting, 1.0);
//}
//
