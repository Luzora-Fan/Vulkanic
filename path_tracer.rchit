#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "path_tracer_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT ShadowPayload shadowPayload;
hitAttributeEXT vec2 hitAttributes;

float SmithLambda(float ax, float ay, vec3 localDirection)
{
    float localDirectionZ2 = localDirection.z * localDirection.z;
    if (localDirectionZ2 <= 1.0e-6)
    {
        return 1.0e8;
    }

    float slope = (ax * ax * localDirection.x * localDirection.x + ay * ay * localDirection.y * localDirection.y)
                  / localDirectionZ2;
    return 0.5 * (-1.0 + sqrt(1.0 + slope));
}

float GgxDistribution(float ax, float ay, vec3 localHalfVector)
{
    if (localHalfVector.z <= 0.0)
    {
        return 0.0;
    }

    float scaledX = localHalfVector.x / ax;
    float scaledY = localHalfVector.y / ay;
    float scaledZ = localHalfVector.z;
    float denominator = scaledX * scaledX + scaledY * scaledY + scaledZ * scaledZ;
    return 1.0 / max(PI * ax * ay * denominator * denominator, 1.0e-6);
}

void main()
{
    InstanceData instanceData = instanceBuffer.instances[gl_InstanceCustomIndexEXT];
    Material material = GetInstanceMaterial(instanceData);
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
    mat3 worldToObjectLinear =
        mat3(gl_WorldToObjectEXT[0].xyz, gl_WorldToObjectEXT[1].xyz, gl_WorldToObjectEXT[2].xyz);
    vec3 objectNormal = GetTriangleObjectNormal(instanceData, gl_PrimitiveID, hitAttributes);
    vec3 normal = TransformNormalToWorld(objectNormal, worldToObjectLinear);
    if (dot(normal, gl_WorldRayDirectionEXT) > 0.0)
    {
        normal = -normal;
    }

    vec3 surfaceThroughput = payload.throughput.xyz * material.albedo;
    if (MaxComponent(surfaceThroughput) < 1.0e-6)
    {
        return;
    }

    RNG rng;
    rng.state = payload.state.x;

    vec3 V = -gl_WorldRayDirectionEXT;
    vec3 helper = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(helper, normal));
    vec3 bitangent = cross(normal, tangent);

    vec3 V_local = normalize(vec3(dot(V, tangent), dot(V, bitangent), dot(V, normal)));
    float a = material.roughness * material.roughness;

    // Optional anisotropy support
    float anisotropy = max(0.0, material.emission.x * 0.0); // Dummy placeholder for anisotropy, typically from material properties
    float ax = max(0.001, a * (1.0 + anisotropy));
    float ay = max(0.001, a * (1.0 - anisotropy));

    float NdotV = max(V_local.z, 0.0);
    float sunConeAngle = max(SkySunRadius(), 1.0e-4);
    vec3 sunDirection = sample_cone(GetSunDirection(), sunConeAngle, rng);
    float NdotL = max(dot(normal, sunDirection), 0.0);
    if (NdotV > 0.0 && NdotL > 0.0)
    {
        shadowPayload.visible = 0u;
        traceRayEXT(topLevelAS,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF,
        0,
        0,
        1,
        OffsetRay(hitPosition, normal),
        0.001,
        sunDirection,
        1e30,
        1);

        if (shadowPayload.visible != 0u)
        {
            vec3 H = V + sunDirection;
            float halfVectorLength2 = dot(H, H);
            if (halfVectorLength2 > 1.0e-6)
            {
                H *= inversesqrt(halfVectorLength2);

                vec3 sunLocal = vec3(dot(sunDirection, tangent), dot(sunDirection, bitangent), NdotL);
                vec3 H_local = vec3(dot(H, tangent), dot(H, bitangent), dot(H, normal));
                float D = GgxDistribution(ax, ay, H_local);
                float LambdaV = SmithLambda(ax, ay, V_local);
                float LambdaL = SmithLambda(ax, ay, sunLocal);
                float G = 1.0 / (1.0 + LambdaV + LambdaL);
                vec3 F = FresnelReflectance(max(dot(H, V), 0.0), material.eta, material.extinction);
                vec3 specularBrdf = F * (D * G / max(4.0 * NdotV * NdotL, 1.0e-6));
                vec3 sunSample = direct_sun_radiance(sunDirection) * sun_solid_angle(sunConeAngle) * NdotL;
                payload.radiance.xyz += surfaceThroughput * specularBrdf * sunSample;
            }
        }
    }

    payload.throughput.xyz = surfaceThroughput;

    if (payload.state.y >= 3u)
    {
        float surviveProbability = clamp(MaxComponent(payload.throughput.xyz), 0.05, 0.95);
        if (NextFloat(rng.state) > surviveProbability)
        {
            payload.state.x = rng.state;
            return;
        }
        payload.throughput.xyz /= surviveProbability;
    }

    payload.state.y += 1u;

    vec2 xi = vec2(NextFloat(rng.state), NextFloat(rng.state));

    // Heitz 2018 VNDF sampling (anisotropic)
    vec3 Vh = normalize(vec3(ax * V_local.x, ay * V_local.y, V_local.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);

    float r = sqrt(xi.x);
    float phi = 2.0 * PI * xi.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    vec3 Ne = normalize(vec3(ax * Nh.x, ay * Nh.y, max(0.0, Nh.z)));

    // Transform back to world space
    vec3 H = normalize(tangent * Ne.x + bitangent * Ne.y + normal * Ne.z);
    vec3 reflectedSample = reflect(-V, H);

    vec3 L_local = normalize(vec3(dot(reflectedSample, tangent), dot(reflectedSample, bitangent), dot(reflectedSample, normal)));

    if (L_local.z <= 0.0)
    {
        payload.throughput.xyz = vec3(0.0);
    }
    else
    {
        float cosThetaH = clamp(dot(H, V), 0.0, 1.0);
        vec3 F = FresnelReflectance(cosThetaH, material.eta, material.extinction);

        float LambdaV = SmithLambda(ax, ay, V_local);
        float LambdaL = SmithLambda(ax, ay, L_local);
        float G2_over_G1 = (1.0 + LambdaV) / (1.0 + LambdaV + LambdaL);

        vec3 single_scattering = F * G2_over_G1;

        // Multiscatter GGX Energy Compensation (Fdez-Aguera approximation)
        float E0 = G2_over_G1;
        vec3 F_avg = F;
        vec3 multi_scattering = F_avg * ((1.0 - E0) * (1.0 - E0)) / (1.0 - F_avg * (1.0 - E0));

        payload.throughput.xyz *= (single_scattering + multi_scattering);
    }

    payload.state.x = rng.state;
    traceRayEXT(topLevelAS,
    gl_RayFlagsOpaqueEXT | gl_RayFlagsCullBackFacingTrianglesEXT,
    0xFF,
    0,
    0,
    0,
    OffsetRay(hitPosition, normal),
    0.0,
    reflectedSample,
    1e30,
    0);
}
