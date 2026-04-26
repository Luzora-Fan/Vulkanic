#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "path_tracer_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;
hitAttributeEXT vec2 hitAttributes;

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

    payload.throughput.xyz *= material.albedo;
    if (MaxComponent(payload.throughput.xyz) < 1.0e-6)
    {
        return;
    }

    RNG rng;
    rng.state = payload.state.x;

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

    vec3 V = -gl_WorldRayDirectionEXT;
    vec3 helper = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(helper, normal));
    vec3 bitangent = cross(normal, tangent);
    
    // Transform V to local space
    vec3 V_local = normalize(vec3(dot(V, tangent), dot(V, bitangent), dot(V, normal)));

    vec2 xi = vec2(NextFloat(rng.state), NextFloat(rng.state));
    float a = material.roughness * material.roughness;

    // Optional anisotropy support
    float anisotropy = max(0.0, material.emission.x * 0.0); // Dummy placeholder for anisotropy, typically from material properties
    float ax = max(0.001, a * (1.0 + anisotropy));
    float ay = max(0.001, a * (1.0 - anisotropy));

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
    t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
    
    vec3 Nh = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2))*Vh;
    vec3 Ne = normalize(vec3(ax * Nh.x, ay * Nh.y, max(0.0, Nh.z)));
    
    // Transform back to world space
    vec3 H = normalize(tangent * Ne.x + bitangent * Ne.y + normal * Ne.z);
    vec3 reflectedSample = reflect(-V, H);
    
    // L in local space
    vec3 L_local = normalize(vec3(dot(reflectedSample, tangent), dot(reflectedSample, bitangent), dot(reflectedSample, normal)));

    if (L_local.z <= 0.0) {
        payload.throughput.xyz = vec3(0.0);
    } else {
        // Fresnel reflection
        float cosThetaH = clamp(dot(H, V), 0.0, 1.0);
        vec3 F = FresnelReflectance(cosThetaH, material.eta, material.extinction);

        // Smith G2/G1 masking weight
        // Anisotropic version
        float LambdaV = 0.5 * (-1.0 + sqrt(1.0 + (ax*ax * V_local.x*V_local.x + ay*ay * V_local.y*V_local.y) / (V_local.z * V_local.z)));
        float LambdaL = 0.5 * (-1.0 + sqrt(1.0 + (ax*ax * L_local.x*L_local.x + ay*ay * L_local.y*L_local.y) / (L_local.z * L_local.z)));
        float G2_over_G1 = (1.0 + LambdaV) / (1.0 + LambdaV + LambdaL);

        // Single-scattering weight
        vec3 single_scattering = F * G2_over_G1;

        // Multiscatter GGX Energy Compensation (Fdez-Aguera approximation)
        // E0 represents the directional albedo for a perfectly white material. 
        // We approximate it directly using the masking weight G2_over_G1.
        float E0 = G2_over_G1;
        vec3 F_avg = F; // Simplified average Fresnel over the hemisphere
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
