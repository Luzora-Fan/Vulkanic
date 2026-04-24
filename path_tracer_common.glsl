#ifndef PATH_TRACER_COMMON_INCLUDED
#define PATH_TRACER_COMMON_INCLUDED

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D outputImage;
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 2) uniform SceneData
{
    vec4 worldToObjectRow0;
    vec4 worldToObjectRow1;
    vec4 worldToObjectRow2;
    vec4 materialAlbedoRoughness;
    vec4 materialEmission;
    vec4 materialEta;
    vec4 materialExtinction;
    vec4 skyBetaRayleighBetaM;
    vec4 skyMieEarthAtmosScaleHr;
    vec4 skyScaleHmSunRadiusAa;
    vec4 skySunRadiance;
    uvec4 skySampleCounts;
} sceneData;

layout(push_constant) uniform PushConstants
{
    vec4 cameraPositionFrame;
    vec4 cameraForwardSamples;
    vec4 cameraRightBounces;
    vec4 cameraUpTanHalfFovY;
    vec4 skyBottomExposure;
    vec4 skyTopAspect;
    uvec2 imageSize;
} pc;

#include "sky.comp"

struct Material
{
    vec3 albedo;
    vec3 emission;
    float roughness;
    vec3 eta;
    vec3 extinction;
};

struct RayPayload
{
    vec4 radiance;
    vec4 throughput;
    uvec4 state;
};

uint Hash(uint x)
{
    x ^= 2747636419u;
    x *= 2654435769u;
    x ^= x >> 16;
    x *= 2654435769u;
    x ^= x >> 16;
    x *= 2654435769u;
    return x;
}

float NextFloat(inout uint state)
{
    state = 1664525u * state + 1013904223u;
    return float(state & 0x00FFFFFFu) / 16777216.0;
}

vec2 NextFloat2(inout uint state)
{
    return vec2(NextFloat(state), NextFloat(state));
}

vec3 SampleSky(vec3 direction, inout uint rngState)
{
    vec3 sunDir = normalize(vec3(0.35, 0.85, 0.25));
    RNG rng;
    rng.state = rngState;
    vec3 sky = render_sky_pixel(direction, sunDir, rng);
    rngState = rng.state;
    return sky * pc.skyBottomExposure.w;
}

vec3 ToneMap(vec3 colour)
{
    colour = max(colour, vec3(0.0));
    colour = colour / (vec3(1.0) + colour);
    return pow(colour, vec3(1.0 / 2.2));
}

float MaxComponent(vec3 value)
{
    return max(value.r, max(value.g, value.b));
}

vec3 FresnelReflectance(float cosThetaI, vec3 eta, vec3 extinction)
{
    cosThetaI = clamp(cosThetaI, 0.0, 1.0);
    vec3 eta2 = eta * eta;
    vec3 extinction2 = extinction * extinction;
    float cosThetaI2 = cosThetaI * cosThetaI;
    vec3 twoEtaCosTheta = 2.0 * eta * cosThetaI;

    vec3 rsNumerator = eta2 + extinction2 - twoEtaCosTheta + vec3(cosThetaI2);
    vec3 rsDenominator = eta2 + extinction2 + twoEtaCosTheta + vec3(cosThetaI2);
    vec3 rpNumerator = (eta2 + extinction2) * cosThetaI2 - twoEtaCosTheta + vec3(1.0);
    vec3 rpDenominator = (eta2 + extinction2) * cosThetaI2 + twoEtaCosTheta + vec3(1.0);

    vec3 rs = rsNumerator / max(rsDenominator, vec3(1.0e-6));
    vec3 rp = rpNumerator / max(rpDenominator, vec3(1.0e-6));
    return clamp(0.5 * (rs + rp), vec3(0.0), vec3(1.0));
}

vec3 TransformPointToObject(vec3 worldPoint)
{
    return vec3(dot(sceneData.worldToObjectRow0.xyz, worldPoint) + sceneData.worldToObjectRow0.w,
                dot(sceneData.worldToObjectRow1.xyz, worldPoint) + sceneData.worldToObjectRow1.w,
                dot(sceneData.worldToObjectRow2.xyz, worldPoint) + sceneData.worldToObjectRow2.w);
}

vec3 TransformNormalToWorld(vec3 objectNormal)
{
    return normalize(vec3(sceneData.worldToObjectRow0.x * objectNormal.x
                          + sceneData.worldToObjectRow1.x * objectNormal.y
                          + sceneData.worldToObjectRow2.x * objectNormal.z,
                          sceneData.worldToObjectRow0.y * objectNormal.x
                          + sceneData.worldToObjectRow1.y * objectNormal.y
                          + sceneData.worldToObjectRow2.y * objectNormal.z,
                          sceneData.worldToObjectRow0.z * objectNormal.x
                          + sceneData.worldToObjectRow1.z * objectNormal.y
                          + sceneData.worldToObjectRow2.z * objectNormal.z));
}

Material GetSphereMaterial()
{
    Material metal;
    metal.albedo = sceneData.materialAlbedoRoughness.xyz;
    metal.emission = sceneData.materialEmission.xyz;
    metal.roughness = sceneData.materialAlbedoRoughness.w;
    metal.eta = sceneData.materialEta.xyz;
    metal.extinction = sceneData.materialExtinction.xyz;
    return metal;
}

#endif
