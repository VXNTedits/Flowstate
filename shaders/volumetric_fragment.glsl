#version 330 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D depthMap;

struct Light {
    vec3 position;
    vec3 color;
};

uniform Light lights[10];  // Assuming a maximum of 10 lights for this example
uniform int lightCount;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 cameraPosition;
uniform float lightScatteringCoefficient;
uniform mat4 inverseLightSpaceMatrix;

float getDepth(vec2 uv) {
    return texture(depthMap, uv).r;
}

void main()
{
    float depth = getDepth(TexCoords);

    // Reconstruct fragment position from depth
    vec4 ndcPosition = vec4(TexCoords * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 fragPositionInLightSpace = inverseLightSpaceMatrix * ndcPosition;
    vec3 fragPosition = fragPositionInLightSpace.xyz / fragPositionInLightSpace.w;

    // Transform frag position into view space
    vec3 fragPositionInViewSpace = (view * vec4(fragPosition, 1.0)).xyz;
    vec3 viewDir = normalize(cameraPosition - fragPositionInViewSpace);

    float totalScattering = 0.0;
    vec3 totalColor = vec3(0.0);

    for (int i = 0; i < lightCount; i++) {
        vec3 lightDirection = normalize(lights[i].position - fragPosition);
        float distance = length(lights[i].position - fragPosition);

        // Compute scattering based on distance and angle to the light source
        float scattering = exp(-lightScatteringCoefficient * distance);
        float angleFactor = max(dot(viewDir, lightDirection), 0.0);  // Ensure non-negative scattering

        totalScattering += scattering * angleFactor;
        totalColor += lights[i].color * scattering * angleFactor;
    }

    // Average the scattering effect and color from all lights
    totalScattering /= float(lightCount);
    totalColor /= float(lightCount);

    // Output the scattering effect and color as the fragment color
    FragColor = vec4(totalScattering * totalColor, 1.0);
}
