//#version 330 core
//in vec2 TexCoords; // Input texture coordinates from the vertex shader
//out vec4 FragColor; // Output fragment color
//
//uniform vec3 camPos; // Camera position in world coordinates
//uniform vec3 volumeMin; // Minimum corner of the volume in world coordinates
//uniform vec3 volumeMax; // Maximum corner of the volume in world coordinates
//uniform int numLights; // Number of lights
//uniform vec3 lightPositions[10]; // Positions of the lights (assuming a maximum of 10 lights)
//uniform vec3 lightColors[10]; // Colors of the lights
//uniform float density; // Density of the fog
//
//void main()
//{
//    // Compute the ray direction from the camera position to the texture coordinates
//    vec3 dir = normalize(vec3(TexCoords, 0.5) - camPos);
//
//    // Initialize the starting position of the ray to the camera position
//    vec3 pos = camPos;
//
//    // Define the step size for ray marching
//    float stepSize = 0.01;
//
//    // Compute the step vector based on the ray direction and step size
//    vec3 step = dir * stepSize;
//
//    // Initialize the accumulated color and alpha values
//    vec4 accumulatedColor = vec4(0.0);
//    float accumulatedAlpha = 0.0;
//
//    // Perform ray marching for a fixed number of steps
//    for (int i = 0; i < 100; i++) {
//        // Move the current position along the ray by the step vector
//        pos += step;
//
//        // Check if the current position is outside the volume bounds
//        if (pos.x < volumeMin.x || pos.y < volumeMin.y || pos.z < volumeMin.z ||
//            pos.x > volumeMax.x || pos.y > volumeMax.y || pos.z > volumeMax.z)
//            break; // Exit the loop if outside the bounds
//
//        // Compute the scattering effect
//        vec3 lighting = vec3(0.0);
//        for (int j = 0; j < numLights; ++j) {
//            vec3 lightDir = normalize(lightPositions[j] - pos);
//            float scatter = max(dot(dir, lightDir), 0.0);
//            lighting += lightColors[j] * scatter;
//        }
//
//        // Compute the color of the current sample, using the constant density
//        vec4 color = vec4(lighting * density, density); // Fog color modulated by light
//
//        // Compute the alpha contribution of the current sample
//        float alpha = color.a * (1.0 - accumulatedAlpha);
//
//        // Accumulate the color and alpha values
//        accumulatedColor.rgb += color.rgb * alpha;
//        accumulatedAlpha += alpha;
//    }
//
//    // Set the final fragment color to the accumulated color and alpha
//    FragColor = accumulatedColor;
//}

#version 330 core
in vec2 TexCoords; // Input texture coordinates from the vertex shader
out vec4 FragColor; // Output fragment color

uniform vec3 camPos; // Camera position in world coordinates
uniform vec3 volumeMin; // Minimum corner of the volume in world coordinates
uniform vec3 volumeMax; // Maximum corner of the volume in world coordinates
uniform int numLights; // Number of lights
uniform vec3 lightPositions[10]; // Positions of the lights (assuming a maximum of 10 lights)
uniform vec3 lightColors[10]; // Colors of the lights
uniform float density; // Density of the fog

void main()
{
    // Compute the ray direction from the camera position to the texture coordinates
    vec3 dir = normalize(vec3(TexCoords, 0.5) - camPos);

    // Initialize the starting position of the ray to the camera position
    vec3 pos = camPos;

    // Define the step size for ray marching
    float stepSize = 0.01;

    // Compute the step vector based on the ray direction and step size
    vec3 step = dir * stepSize;

    // Initialize the accumulated color and alpha values
    vec4 accumulatedColor = vec4(0.0);
    float accumulatedAlpha = 0.0;

    // Perform ray marching for a fixed number of steps
    for (int i = 0; i < 100; i++) {
        // Move the current position along the ray by the step vector
        pos += step;

        // Check if the current position is outside the volume bounds
        if (pos.x < volumeMin.x || pos.y < volumeMin.y || pos.z < volumeMin.z ||
            pos.x > volumeMax.x || pos.y > volumeMax.y || pos.z > volumeMax.z)
            break; // Exit the loop if outside the bounds

        // Compute the scattering effect
        vec3 lighting = vec3(0.0);
        for (int j = 0; j < numLights; ++j) {
            vec3 lightDir = normalize(lightPositions[j] - pos);
            float scatter = max(dot(dir, lightDir), 0.0);
            lighting += lightColors[j] * scatter * (1.0 / (j + 1)); // Modulate color by light index
        }

        // Compute the color of the current sample, using the constant density
        vec4 color = vec4(lighting * density, density); // Fog color modulated by light

        // Compute the alpha contribution of the current sample
        float alpha = color.a * (1.0 - accumulatedAlpha);

        // Accumulate the color and alpha values
        accumulatedColor.rgb += color.rgb * alpha;
        accumulatedAlpha += alpha;
    }

    // Set the final fragment color to the accumulated color and alpha
    FragColor = accumulatedColor;
}
