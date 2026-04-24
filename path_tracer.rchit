#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "path_tracer_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;
hitAttributeEXT vec2 hitAttributes;

void main()
{
    Material material = GetSphereMaterial();
    payload.radiance.xyz += payload.throughput.xyz * material.emission;
    if (MaxComponent(material.emission) > 0.0)
    {
        return;
    }

    uint maxBounces = uint(pc.cameraRightBounces.w);
    if (payload.state.y + 1u >= maxBounces)
    {
        return;
    }

    vec3 hitPosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 objectPosition = TransformPointToObject(hitPosition);
    vec3 normal = TransformNormalToWorld(normalize(objectPosition));
    if (dot(normal, gl_WorldRayDirectionEXT) > 0.0)
    {
        normal = -normal;
    }

    float cosTheta = clamp(dot(normal, -gl_WorldRayDirectionEXT), 0.0, 1.0);
    payload.throughput.xyz *= material.albedo * FresnelReflectance(cosTheta, material.eta, material.extinction);
    if (MaxComponent(payload.throughput.xyz) < 1.0e-6)
    {
        return;
    }

    payload.state.y += 1u;

    if (payload.state.y >= 2u)
    {
        uint rngState = payload.state.x;
        float surviveProbability = clamp(MaxComponent(payload.throughput.xyz), 0.05, 0.95);
        if (NextFloat(rngState) > surviveProbability)
        {
            payload.state.x = rngState;
            return;
        }
        payload.throughput.xyz /= surviveProbability;
        payload.state.x = rngState;
    }

    RNG rng;
    rng.state = payload.state.x;
    vec3 reflectedDirection = reflect(gl_WorldRayDirectionEXT, normal);
    vec3 reflectedSample = sample_cone(reflectedDirection, material.roughness * 0.5 * PI, rng);
    payload.state.x = rng.state;
    traceRayEXT(topLevelAS,
    gl_RayFlagsOpaqueEXT,
    0xFF,
    0,
    0,
    0,
    hitPosition + normal * 0.001,
    0.001,
    reflectedSample,
    1e30,
    0);
}
