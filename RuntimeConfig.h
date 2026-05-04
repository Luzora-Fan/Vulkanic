#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

inline constexpr float kPi = 3.14159265358979323846f;

struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

inline bool operator==(const Vec3& left, const Vec3& right)
{
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

inline bool operator!=(const Vec3& left, const Vec3& right)
{
    return !(left == right);
}

inline Vec3 operator+(const Vec3& left, const Vec3& right)
{
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

inline Vec3 operator-(const Vec3& left, const Vec3& right)
{
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

inline Vec3 operator*(const Vec3& value, float scalar)
{
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

inline Vec3& operator+=(Vec3& left, const Vec3& right)
{
    left.x += right.x;
    left.y += right.y;
    left.z += right.z;
    return left;
}

inline float Length(const Vec3& value)
{
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

inline Vec3 Normalize(const Vec3& value)
{
    const float length = Length(value);
    if (length <= 0.0f)
    {
        return {};
    }
    return value * (1.0f / length);
}

inline Vec3 Cross(const Vec3& left, const Vec3& right)
{
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

struct MaterialConfig
{
    std::array<float, 3> albedo{1.0f, 1.0f, 1.0f};
    std::array<float, 3> emission{0.0f, 0.0f, 0.0f};
    float roughness = 0.0f;
    std::array<float, 3> eta{1.5f, 1.5f, 1.5f};
    std::array<float, 3> extinction{1.5f, 1.5f, 1.5f};
};

inline bool operator==(const MaterialConfig& left, const MaterialConfig& right)
{
    return left.albedo == right.albedo && left.emission == right.emission && left.roughness == right.roughness
           && left.eta == right.eta && left.extinction == right.extinction;
}

inline bool operator!=(const MaterialConfig& left, const MaterialConfig& right)
{
    return !(left == right);
}

struct ModelAssetConfig
{
    std::string fileName = "box.obj";
};

inline bool operator==(const ModelAssetConfig& left, const ModelAssetConfig& right)
{
    return left.fileName == right.fileName;
}

inline bool operator!=(const ModelAssetConfig& left, const ModelAssetConfig& right)
{
    return !(left == right);
}

struct ModelInstanceConfig
{
    Vec3 position{};
    Vec3 rotationDegrees{};
    Vec3 scale{1.0f, 1.0f, 1.0f};
    uint32_t modelIndex = 0;
    uint32_t materialIndex = 0;
};

inline bool operator==(const ModelInstanceConfig& left, const ModelInstanceConfig& right)
{
    return left.position == right.position && left.rotationDegrees == right.rotationDegrees && left.scale == right.scale
           && left.modelIndex == right.modelIndex && left.materialIndex == right.materialIndex;
}

inline bool operator!=(const ModelInstanceConfig& left, const ModelInstanceConfig& right)
{
    return !(left == right);
}

struct SkySpectralConfig
{
    std::array<float, 3> betaRayleigh{3.8e-6f, 13.5e-6f, 33.1e-6f};
    float betaMie = 21e-6f;
    float mieG = 0.76f;
    float earthRadius = 6360e3f;
    float atmosphereRadius = 6420e3f;
    float scaleHeightRayleigh = 7994.0f;
    float scaleHeightMie = 1200.0f;
    std::array<float, 3> sunRadiance{20.0f, 18.0f, 14.5f};
    std::array<float, 3> sunDirection{0.35f, 0.3f, 0.25f};
    float sunRadius = 0.1f;
    float sunAa = 0.01f;
    uint32_t secondarySamples = 1;
    uint32_t viewSteps = 5;
    uint32_t samples = 1;
};

struct RuntimeConfig
{
    uint32_t width = 960;
    uint32_t height = 540;
    uint32_t frameCount = 2;
    uint32_t samplesPerPixel = 1;
    uint32_t maxBounces = 2;
    Vec3 initialPosition{0.0f, 0.35f, -6.5f};
    Vec3 initialLookAt{0.0f, -0.1f, 3.8f};
    float fovYDegrees = 40.0f;
    float moveSpeed = 4.0f;
    float fastMoveSpeed = 10.0f;
    float mouseSensitivity = 0.0035f;
    float keyLookSpeed = 1.8f;
    float maxPitchDegrees = 89.0f;
    std::array<float, 3> skyBottomColor{0.03f, 0.04f, 0.06f};
    float skyExposure = 1.35f;
    std::array<float, 3> skyTopColor{0.55f, 0.72f, 0.95f};
    SkySpectralConfig skySpectral{};
    std::vector<ModelAssetConfig> models{ModelAssetConfig{}};
    std::vector<MaterialConfig> materials{MaterialConfig{}};
    std::vector<ModelInstanceConfig> instances{ModelInstanceConfig{}};
};

std::filesystem::path ResolveRuntimeFilePath(const wchar_t* fileName);
std::string LoadTextFile(const std::filesystem::path& filePath);
RuntimeConfig ParseRuntimeConfig(const std::string& jsonText);
