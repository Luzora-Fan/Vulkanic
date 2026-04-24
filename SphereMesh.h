#pragma once

#include <array>
#include <cstdint>
#include <vector>

struct SphereMesh
{
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
};

SphereMesh CreateSphereMesh(const std::array<float, 3>& center,
                            float radius,
                            uint32_t longitudeSegments,
                            uint32_t latitudeSegments);
