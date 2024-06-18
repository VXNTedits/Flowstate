#version 330 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler3D volumeData;
uniform mat4 invViewProjMatrix;
uniform mat4 viewMatrix;
uniform vec3 volumeMin;
uniform vec3 volumeMax;
uniform int numLights;
uniform vec3 lightPositions[10]; // Assuming a maximum of 10 lights
uniform vec3 lightColors[10];

void main()
{
    // Compute the ray direction
    vec4 clipPos = vec4(TexCoords * 2.0 - 1.0, 1.0, 1.0);
    vec4 viewPos = invViewProjMatrix * clipPos;
    vec3 rayDir = normalize(viewPos.xyz / viewPos.w);

    // Compute the camera position
    vec3 camPos = vec3(inverse(viewMatrix) * vec4(0.0, 0.0, 0.0, 1.0));

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
    vec4 color = vec4(0.0);
    float alpha_accum = 0.0;

    for (float t = tMin; t < tMax; t += 0.1)
    {
        vec3 samplePos = (pos - volumeMin) / (volumeMax - volumeMin); // Transform to [0, 1]
        float sampleValue = texture(volumeData, samplePos).r; // Sample red channel as alpha
        vec4 sampleColor = vec4(sampleValue, sampleValue, sampleValue, sampleValue); // Use sampled value as alpha and grayscale color

        // Apply lighting from all lights
        vec3 lighting = vec3(0.0);
        for (int i = 0; i < numLights; ++i)
        {
            vec3 lightDir = normalize(lightPositions[i] - pos);
            float diff = max(dot(rayDir, lightDir), 0.0);
            lighting += lightColors[i] * diff;
        }
        sampleColor.rgb *= lighting;

        // Accumulate color and alpha
        float alpha = sampleColor.a * (1.0 - alpha_accum);
        color.rgb += sampleColor.rgb * alpha;
        alpha_accum += alpha;

        pos += step;
        if (alpha_accum >= 0.5) break; // Early exit if fully opaque
    }

    // Final color composition
    FragColor = vec4(color.rgb, alpha_accum);
}
