#version 430 core

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D sceneColor;          // The texture containing the rendered scene colors
uniform sampler2D sceneDepth;          // The texture containing the rendered scene depth
uniform vec3 fogColor;                 // The color of the fog
uniform float fogDensity;              // The density of the fog
uniform float fogHeightFalloff;        // The rate at which fog density decreases with height
uniform mat4 invProj;                  // The inverse projection matrix
uniform float near;                    // The near clipping plane distance
uniform float far;                     // The far clipping plane distance

// Convert non-linear depth value to linear depth value
float getLinearDepth(float depth)
{
    float z = depth * 2.0 - 1.0; // Convert depth from [0,1] range to Normalized Device Coordinates (NDC) range [-1, 1]
    return (2.0 * near * far) / (far + near - z * (far - near)); // Convert NDC depth to linear depth using near and far plane values
}

void main()
{
    vec4 color = texture(sceneColor, TexCoords); // Fetch the color of the current fragment from the scene color texture
    float depth = texture(sceneDepth, TexCoords).r; // Fetch the depth of the current fragment from the scene depth texture
    //float linearDepth = getLinearDepth(depth); // Convert the non-linear depth to linear depth

    // Reconstruct world position from depth
    vec4 clipSpacePos = vec4((TexCoords * 2.0 - 1.0), depth, 1.0); // Reconstruct the clip space position using TexCoords and depth
    vec4 viewSpacePos = invProj * clipSpacePos; // Transform clip space position to view space position using the inverse projection matrix
    viewSpacePos /= viewSpacePos.w; // Perform perspective division to get the actual view space position

    // Use the y-component of the view space position for fog calculation
    float fogFactor = exp(-fogDensity * (1.0 - exp(-viewSpacePos.y * fogHeightFalloff))); // Calculate the fog factor based on the height (y) of the fragment in view space
    fogFactor = clamp(fogFactor, 0.0, 1.0); // Clamp the fog factor to the range [0, 1]

    vec3 finalColor = mix(fogColor, color.rgb, fogFactor); // Mix the original color with the fog color based on the fog factor
    FragColor = vec4(finalColor, 1.0); // Set the final color of the fragment
}
