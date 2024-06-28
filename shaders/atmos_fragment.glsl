#version 430 core

in vec3 TexCoord;  // Ensure TexCoord is in the correct coordinate system

out vec4 FragColor;

uniform sampler3D fogTexture;
uniform vec3 lightPosition;
uniform vec3 lightColor;
uniform float lightIntensity;
uniform vec3 cameraPosition;
uniform float scatteringCoefficient;
uniform mat4 viewMatrix;  // Camera view matrix
uniform mat4 projectionMatrix;  // Camera projection matrix

const vec3 scatteringColor = vec3(0.5, 0.7, 1.0); // Typical blue tint for Rayleigh scattering

void main()
{
    // Transform TexCoord to world space coordinates
    vec4 worldCoord = inverse(viewMatrix) * vec4(TexCoord, 1.0);

    // Compute the vector from the camera to the fragment position for view-dependent effects
    vec3 viewDir = normalize(cameraPosition - worldCoord.xyz);

    // Compute the light direction from the light position to the fragment position
    vec3 lightDir = normalize(lightPosition - worldCoord.xyz);

    // Sample the fog density from the 3D texture at the fragment's position
    float density = texture(fogTexture, TexCoord).r;

    // Calculate the distance from the camera to the fragment position
    float distance = length(cameraPosition - worldCoord.xyz);

    // Calculate the Rayleigh scattering effect
    float scatter = exp(-scatteringCoefficient * density * distance);

    // Rayleigh Phase Function component
    float cosTheta = dot(viewDir, lightDir);
    float rayleighPhase = 0.75 * (1.0 + cosTheta * cosTheta);

    // Calculate the color of the light scattered with Rayleigh phase function
    vec3 scatteredLight = lightColor * scatteringColor * scatter * rayleighPhase;

    // Integrate light intensity and fog density into the final color
    vec3 color = scatteredLight * lightIntensity * density;

    // Compute alpha value based on distance to simulate increasing atmospheric effect with distance
    float alpha = clamp(distance * scatteringCoefficient * 0.5, 0.7, 1.0); // Adjust the multiplier as needed

    // Output the final color with variable alpha
    FragColor = vec4(color, alpha);
}
