#include "SphereMesh.h"

#include "RuntimeConfig.h"

#include <stdexcept>

SphereMesh CreateSphereMesh(const std::array<float, 3>& center,
                            float radius,
                            uint32_t longitudeSegments,
                            uint32_t latitudeSegments)
{
    if (longitudeSegments < 3 || latitudeSegments < 2)
    {
        throw std::runtime_error("Sphere mesh requires at least 3 longitude and 2 latitude segments.");
    }

    SphereMesh mesh{};
    const uint32_t ringCount = latitudeSegments - 1;
    mesh.vertices.reserve(static_cast<size_t>(2 + ringCount * longitudeSegments) * 3);
    mesh.indices.reserve(static_cast<size_t>(longitudeSegments) * 6 * static_cast<size_t>(latitudeSegments - 1));

    auto appendVertex = [&mesh](float x, float y, float z)
    {
        mesh.vertices.push_back(x);
        mesh.vertices.push_back(y);
        mesh.vertices.push_back(z);
    };

    appendVertex(center[0], center[1] + radius, center[2]);

    for (uint32_t latitude = 1; latitude < latitudeSegments; ++latitude)
    {
        const float v = static_cast<float>(latitude) / static_cast<float>(latitudeSegments);
        const float phi = v * kPi;
        const float y = std::cos(phi) * radius;
        const float ringRadius = std::sin(phi) * radius;

        for (uint32_t longitude = 0; longitude < longitudeSegments; ++longitude)
        {
            const float u = static_cast<float>(longitude) / static_cast<float>(longitudeSegments);
            const float theta = u * 2.0f * kPi;
            appendVertex(center[0] + std::cos(theta) * ringRadius,
                         center[1] + y,
                         center[2] + std::sin(theta) * ringRadius);
        }
    }

    const uint32_t bottomIndex = static_cast<uint32_t>(mesh.vertices.size() / 3);
    appendVertex(center[0], center[1] - radius, center[2]);

    for (uint32_t longitude = 0; longitude < longitudeSegments; ++longitude)
    {
        const uint32_t current = 1 + longitude;
        const uint32_t next = 1 + (longitude + 1) % longitudeSegments;
        mesh.indices.push_back(0);
        mesh.indices.push_back(next);
        mesh.indices.push_back(current);
    }

    for (uint32_t ring = 0; ring + 1 < ringCount; ++ring)
    {
        const uint32_t ringStart = 1 + ring * longitudeSegments;
        const uint32_t nextRingStart = ringStart + longitudeSegments;
        for (uint32_t longitude = 0; longitude < longitudeSegments; ++longitude)
        {
            const uint32_t current = ringStart + longitude;
            const uint32_t next = ringStart + (longitude + 1) % longitudeSegments;
            const uint32_t below = nextRingStart + longitude;
            const uint32_t belowNext = nextRingStart + (longitude + 1) % longitudeSegments;

            mesh.indices.push_back(current);
            mesh.indices.push_back(next);
            mesh.indices.push_back(below);

            mesh.indices.push_back(next);
            mesh.indices.push_back(belowNext);
            mesh.indices.push_back(below);
        }
    }

    const uint32_t lastRingStart = 1 + (ringCount - 1) * longitudeSegments;
    for (uint32_t longitude = 0; longitude < longitudeSegments; ++longitude)
    {
        const uint32_t current = lastRingStart + longitude;
        const uint32_t next = lastRingStart + (longitude + 1) % longitudeSegments;
        mesh.indices.push_back(current);
        mesh.indices.push_back(next);
        mesh.indices.push_back(bottomIndex);
    }

    return mesh;
}
