#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "path_tracer_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main()
{
    uint rngState = payload.state.x;
    payload.radiance.xyz += payload.throughput.xyz * SampleSky(gl_WorldRayDirectionEXT, rngState);
    payload.state.x = rngState;
}
