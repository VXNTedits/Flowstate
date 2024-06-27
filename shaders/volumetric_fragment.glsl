#version 430 core

in vec2 TexCoords;
out vec4 FragColor;

uniform mat4 invViewProjMatrix;
uniform vec3 volumeMin;
uniform vec3 volumeMax;
uniform int numLights;
uniform vec3 lightPositions[10]; // Assuming a maximum of 10 lights
uniform vec3 lightColors[10];
uniform float glowIntensity; // Parameter for controlling glow intensity
uniform float scatteringFactor; // Parameter for controlling scattering
uniform float glowFalloff; // Parameter for controlling the sharpness of the glow falloff
uniform float godRayIntensity; // Parameter for controlling god ray intensity
uniform float godRayDecay; // Parameter for controlling god ray decay
uniform float godRaySharpness; // Parameter for controlling god ray sharpness
uniform float time; // Time variable for animated noise

// Simple 3D noise function
float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

float noise(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);

    float n = p.x + p.y * 57.0 + 113.0 * p.z;

    return mix(mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
                   mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y),
               mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                   mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
}

void main()
{
    // Compute the ray direction in world space
    vec4 clipPos = vec4(TexCoords * 2.0 - 1.0, -1.0, 1.0); // Note: using -1.0 for z to start ray from near plane
    vec4 viewPos = invViewProjMatrix * clipPos;
    vec3 rayDir = normalize(viewPos.xyz / viewPos.w);

    // Compute the camera position in world space
    vec3 camPos = vec3(inverse(invViewProjMatrix) * vec4(0.0, 0.0, 0.0, 1.0));

    // Ray-box intersection (volume bounds)
    vec3 invDir = 1.0 / rayDir;
    vec3 t0s = (volumeMin - camPos) * invDir;
    vec3 t1s = (volumeMax - camPos) * invDir;
    float tMin = max(max(min(t0s.x, t1s.x), min(t0s.y, t1s.y)), min(t0s.z, t1s.z));
    float tMax = min(min(max(t0s.x, t1s.x), max(t0s.y, t1s.y)), max(t0s.z, t1s.z));

    // Early exit if no intersection
    if (tMax < tMin) discard;

    // Adjust tMin if the ray starts inside the volume
    if (tMin < 0.0) tMin = 0.0;

    // Ray marching
    vec3 pos = camPos + rayDir * tMin;
    vec3 step = rayDir * 0.001; // Step size
    vec4 scatteredLight = vec4(0.0);
    vec4 transmittance = vec4(1.0);

    for (float t = tMin; t < tMax; t += 0.1)
    {
        vec3 samplePos = (pos - volumeMin) / (volumeMax - volumeMin); // Transform to [0, 1]

        // Animated noise
        float density = noise(samplePos * 5.0 + vec3(0.0, 0.0, time));

        // Light scattering and absorption
        vec3 scattering = vec3(0.0);
        for (int i = 0; i < numLights; ++i)
        {
            vec3 lightDir = normalize(lightPositions[i] - pos);
            float distanceToLight = length(lightPositions[i] - pos);
            float lightIntensity = max(dot(rayDir, lightDir), 0.0); // Simple scattering model

            // Sharper glow falloff
            float glow = exp(-pow(distanceToLight * glowFalloff, 2.0)) * lightIntensity; // Use glowFalloff

            // God ray effect with added sharpness
            float godRayEffect = pow(max(dot(rayDir, lightDir), 0.0), godRayIntensity) * exp(-distanceToLight * godRayDecay);
            godRayEffect = pow(godRayEffect, godRaySharpness); // Increase sharpness

            scattering += lightColors[i] * (lightIntensity + glow * glowIntensity + godRayEffect);
        }

        // Compute attenuation
        vec4 attenuation = vec4(exp(-density * scatteringFactor)); // Use scatteringFactor
        transmittance *= attenuation;

        // Accumulate scattered light
        scatteredLight += vec4(scattering, 1.0) * transmittance * density * 0.1;

        pos += step;
        if (transmittance.a < 0.01) break; // Early exit if nearly fully attenuated
    }

    // Final color composition
    FragColor = vec4(clamp(scatteredLight.rgb, 0.0, 1.0), 1.0 - transmittance.a);
}