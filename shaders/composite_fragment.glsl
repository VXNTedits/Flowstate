#version 430 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D sceneTexture;
uniform sampler2D volumetricTexture;
uniform sampler2D tracers;
uniform sampler3D atmosphere;

uniform sampler2D depthTexture;
uniform float nearPlane;
uniform float farPlane;

float linearizeDepth(float depth)
{
    // Converts depth from [0, 1] range to linear depth value based on near and far planes
    float z = depth * 2.0 - 1.0; // Convert to NDC coordinates
    return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));
}

float computeDepth(vec2 texCoords)
{
    // Sample the depth from the depth texture
    float depth = texture(depthTexture, texCoords).r;

    // Linearize the depth value
    float linearDepth = linearizeDepth(depth);

    // Normalize the linear depth value to [0, 1] range
    float normalizedDepth = (linearDepth - nearPlane) / (farPlane - nearPlane);

    return normalizedDepth;
}

void main()
{
    vec4 sceneColor = texture(sceneTexture, TexCoords);
    vec4 volumetricColor = texture(volumetricTexture, TexCoords);
    vec4 tracerColor = texture(tracers, TexCoords);

    float depth = computeDepth(TexCoords);
    vec3 atmosphereCoords = vec3(TexCoords, depth);
    vec4 atmosphereColor = texture(atmosphere, atmosphereCoords);

    // Combining the colors
    vec4 finalColor = sceneColor + tracerColor + atmosphereColor * atmosphereColor.a + volumetricColor * volumetricColor.a;

    FragColor = finalColor;
}
//#version 430 core
//in vec2 TexCoords;
//out vec4 FragColor;
//
//uniform sampler2D sceneTexture;
//uniform sampler2D volumetricTexture;
//uniform sampler2D tracers;
//uniform sampler3D atmosphere;
//
//uniform sampler2D depthTexture;
//uniform float nearPlane;
//uniform float farPlane;
//
//float linearizeDepth(float depth)
//{
//    // Converts depth from [0, 1] range to linear depth value based on near and far planes
//    float z = depth * 2.0 - 1.0; // Convert to NDC coordinates
//    return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));
//}
//
//float computeDepth(vec2 texCoords)
//{
//    // Sample the depth from the depth texture
//    float depth = texture(depthTexture, texCoords).r;
//
//    // Linearize the depth value
//    float linearDepth = linearizeDepth(depth);
//
//    // Normalize the linear depth value to [0, 1] range
//    float normalizedDepth = (linearDepth - nearPlane) / (farPlane - nearPlane);
//
//    return normalizedDepth;
//}
//
//void main()
//{
//    vec4 sceneColor = texture(sceneTexture, TexCoords);
//    vec4 volumetricColor = texture(volumetricTexture, TexCoords);
//    vec4 tracerColor = texture(tracers, TexCoords);
//
//    float depth = computeDepth(TexCoords);// Compute normalized depth
//    vec3 atmosphereCoords = vec3(TexCoords, depth); // Assuming depth is mapped correctly to [0, 1]
//    vec4 atmosphereColor = texture(atmosphere, atmosphereCoords);
//
//    // Visualize depth information as grayscale
//    vec4 depthColor = vec4(vec3(depth), 1.0); // Convert depth to grayscale
//
//    // Display the depth information for debugging
//    FragColor = depthColor;
//}
