#version 330 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D depthMap;
uniform vec3 lightPositions[10];  // Assuming a maximum of 10 lights for this example
uniform int lightCount;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 cameraPosition;
uniform float lightScatteringCoefficient;

// Add the inverse matrices if you have them precomputed
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

    float totalScattering = 0.0;
    for (int i = 0; i < lightCount; i++) {
        vec3 lightDirection = normalize(lightPositions[i] - fragPosition);
        float scattering = exp(-lightScatteringCoefficient * length(lightDirection));
        totalScattering += scattering;
    }

    // Average the scattering effect from all lights
    totalScattering /= float(lightCount);

    // Output fragPosition for debugging
    FragColor = vec4(fragPosition / 10.0, 1.0);

    //FragColor = vec4(totalScattering, totalScattering, totalScattering, 1.0);
}
