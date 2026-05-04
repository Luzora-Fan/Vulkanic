#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "path_tracer_common.glsl"

layout(location = 1) rayPayloadInEXT ShadowPayload shadowPayload;

void main()
{
    shadowPayload.visible = 1u;
}
