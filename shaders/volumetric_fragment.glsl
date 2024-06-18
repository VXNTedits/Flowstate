#version 330 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler3D volumeData;
uniform mat4 invViewProjMatrix;
uniform vec3 volumeMin;
uniform vec3 volumeMax;
uniform int numLights;
uniform vec3 lightPositions[10]; // Assuming a maximum of 10 lights
uniform vec3 lightColors[10];

void main()
{
    // Compute the ray direction in view space
    vec4 clipPos = vec4(TexCoords * 2.0 - 1.0, 1.0, 1.0);
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

    // Ray marching
    vec3 pos = camPos + rayDir * tMin;
    vec3 step = rayDir * 0.1; // Step size
    vec4 scatteredLight = vec4(0.0);
    vec4 transmittance = vec4(1.0);

    for (float t = tMin; t < tMax; t += 0.1)
    {
        vec3 samplePos = (pos - volumeMin) / (volumeMax - volumeMin); // Transform to [0, 1]
        float density = texture(volumeData, samplePos).r;

        // Light scattering and absorption
        vec3 scattering = vec3(0.0);
        for (int i = 0; i < numLights; ++i)
        {
            vec3 lightDir = normalize(lightPositions[i] - pos);
            float distanceToLight = length(lightPositions[i] - pos);
            float lightIntensity = max(dot(rayDir, lightDir), 0.0); // Simple scattering model
            float glow = exp(-distanceToLight * 0.5) * lightIntensity;

            scattering += lightColors[i] * (lightIntensity + glow);
        }

        // Compute attenuation
        vec4 attenuation = vec4(exp(-density * 0.5));
        transmittance *= attenuation;

        // Accumulate scattered light
        scatteredLight += vec4(scattering, 1.0) * transmittance * density * 0.1;

        pos += step;
        if (transmittance.a < 0.01) break; // Early exit if nearly fully attenuated
    }

    // Final color composition
    FragColor = vec4(clamp(scatteredLight.rgb, 0.0, 1.0), 1.0 - transmittance.a);
}
