#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "RuntimeConfig.h"

#include <windows.h>

#include <array>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string_view>

std::filesystem::path ResolveRuntimeFilePath(const wchar_t* fileName)
{
    WCHAR exePath[MAX_PATH]{};
    const DWORD pathLen = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    if (pathLen == 0 || pathLen == MAX_PATH)
    {
        throw std::runtime_error("Failed to resolve executable path.");
    }

    const auto exeDir = std::filesystem::path(exePath).parent_path();
    const std::array<std::filesystem::path, 3> candidatePaths = {
        exeDir / fileName,
        exeDir.parent_path() / fileName,
        std::filesystem::current_path() / fileName,
    };

    for (const auto& candidate : candidatePaths)
    {
        if (std::filesystem::exists(candidate))
        {
            return candidate;
        }
    }

    return {};
}

std::string LoadTextFile(const std::filesystem::path& filePath)
{
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Failed to open text file.");
    }

    const auto fileSize = file.tellg();
    if (fileSize <= 0)
    {
        throw std::runtime_error("Text file is empty.");
    }

    std::string data(static_cast<size_t>(fileSize), '\0');
    file.seekg(0, std::ios::beg);
    file.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!file)
    {
        throw std::runtime_error("Failed to read text file.");
    }

    return data;
}

static size_t SkipJsonWhitespace(std::string_view text, size_t position)
{
    while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position])))
    {
        ++position;
    }
    return position;
}

static std::optional<size_t> FindJsonValueStart(std::string_view text, const char* key)
{
    const std::string needle = "\"" + std::string(key) + "\"";
    const size_t keyPosition = text.find(needle);
    if (keyPosition == std::string_view::npos)
    {
        return std::nullopt;
    }

    const size_t colonPosition = SkipJsonWhitespace(text, keyPosition + needle.size());
    if (colonPosition >= text.size() || text[colonPosition] != ':')
    {
        throw std::runtime_error("Invalid JSON syntax near a config key.");
    }

    return SkipJsonWhitespace(text, colonPosition + 1);
}

static float ParseJsonNumber(std::string_view text, size_t position, const char* key)
{
    const char* begin = text.data() + position;
    char* end = nullptr;
    const float value = std::strtof(begin, &end);
    if (end == begin)
    {
        throw std::runtime_error(std::string("Expected a numeric value for \"") + key + "\".");
    }
    return value;
}

static uint32_t ParseJsonUint32(std::string_view text, size_t position, const char* key)
{
    const char* begin = text.data() + position;
    char* end = nullptr;
    const unsigned long value = std::strtoul(begin, &end, 10);
    if (end == begin)
    {
        throw std::runtime_error(std::string("Expected an unsigned integer value for \"") + key + "\".");
    }
    if (value > 0xFFFFFFFFul)
    {
        throw std::runtime_error(std::string("Value for \"") + key + "\" exceeds uint32 range.");
    }
    return static_cast<uint32_t>(value);
}

static std::array<float, 3> ParseJsonFloat3(std::string_view text, size_t position, const char* key)
{
    position = SkipJsonWhitespace(text, position);
    if (position >= text.size() || text[position] != '[')
    {
        throw std::runtime_error(std::string("Expected an array value for \"") + key + "\".");
    }

    std::array<float, 3> result{};
    ++position;
    for (size_t i = 0; i < result.size(); ++i)
    {
        position = SkipJsonWhitespace(text, position);
        const char* begin = text.data() + position;
        char* end = nullptr;
        result[i] = std::strtof(begin, &end);
        if (end == begin)
        {
            throw std::runtime_error(std::string("Expected a numeric array element for \"") + key + "\".");
        }

        position = SkipJsonWhitespace(text, static_cast<size_t>(end - text.data()));
        if (i + 1 < result.size())
        {
            if (position >= text.size() || text[position] != ',')
            {
                throw std::runtime_error(std::string("Expected a comma in array value for \"") + key + "\".");
            }
            ++position;
        }
    }

    position = SkipJsonWhitespace(text, position);
    if (position >= text.size() || text[position] != ']')
    {
        throw std::runtime_error(std::string("Expected a closing bracket for \"") + key + "\".");
    }

    return result;
}

static void ParseOptionalJsonNumber(std::string_view text, const char* key, float& target)
{
    const auto valueStart = FindJsonValueStart(text, key);
    if (!valueStart.has_value())
    {
        return;
    }
    target = ParseJsonNumber(text, valueStart.value(), key);
}

static void ParseOptionalJsonUint32(std::string_view text, const char* key, uint32_t& target)
{
    const auto valueStart = FindJsonValueStart(text, key);
    if (!valueStart.has_value())
    {
        return;
    }
    target = ParseJsonUint32(text, valueStart.value(), key);
}

static void ParseOptionalJsonVec3(std::string_view text, const char* key, Vec3& target)
{
    const auto valueStart = FindJsonValueStart(text, key);
    if (!valueStart.has_value())
    {
        return;
    }

    const auto values = ParseJsonFloat3(text, valueStart.value(), key);
    target = {values[0], values[1], values[2]};
}

static void ParseOptionalJsonFloat3(std::string_view text, const char* key, std::array<float, 3>& target)
{
    const auto valueStart = FindJsonValueStart(text, key);
    if (!valueStart.has_value())
    {
        return;
    }
    target = ParseJsonFloat3(text, valueStart.value(), key);
}

static bool HasNegativeElement(const std::array<float, 3>& value)
{
    return value[0] < 0.0f || value[1] < 0.0f || value[2] < 0.0f;
}

RuntimeConfig ParseRuntimeConfig(const std::string& jsonText)
{
    RuntimeConfig config{};
    const std::string_view jsonView(jsonText);

    ParseOptionalJsonUint32(jsonView, "width", config.width);
    ParseOptionalJsonUint32(jsonView, "height", config.height);
    ParseOptionalJsonUint32(jsonView, "frameCount", config.frameCount);
    ParseOptionalJsonUint32(jsonView, "samplesPerPixel", config.samplesPerPixel);
    ParseOptionalJsonUint32(jsonView, "maxBounces", config.maxBounces);
    ParseOptionalJsonVec3(jsonView, "initialPosition", config.initialPosition);
    ParseOptionalJsonVec3(jsonView, "initialLookAt", config.initialLookAt);
    ParseOptionalJsonNumber(jsonView, "fovYDegrees", config.fovYDegrees);
    ParseOptionalJsonNumber(jsonView, "moveSpeed", config.moveSpeed);
    ParseOptionalJsonNumber(jsonView, "fastMoveSpeed", config.fastMoveSpeed);
    ParseOptionalJsonNumber(jsonView, "mouseSensitivity", config.mouseSensitivity);
    ParseOptionalJsonNumber(jsonView, "keyLookSpeed", config.keyLookSpeed);
    ParseOptionalJsonNumber(jsonView, "maxPitchDegrees", config.maxPitchDegrees);
    ParseOptionalJsonFloat3(jsonView, "bottomColor", config.skyBottomColor);
    ParseOptionalJsonNumber(jsonView, "exposure", config.skyExposure);
    ParseOptionalJsonFloat3(jsonView, "topColor", config.skyTopColor);
    ParseOptionalJsonFloat3(jsonView, "BETA_R", config.skySpectral.betaRayleigh);
    ParseOptionalJsonNumber(jsonView, "BETA_M", config.skySpectral.betaMie);
    ParseOptionalJsonNumber(jsonView, "MIE_G", config.skySpectral.mieG);
    ParseOptionalJsonNumber(jsonView, "EARTH_R", config.skySpectral.earthRadius);
    ParseOptionalJsonNumber(jsonView, "ATMOS_R", config.skySpectral.atmosphereRadius);
    ParseOptionalJsonNumber(jsonView, "SCALE_H_R", config.skySpectral.scaleHeightRayleigh);
    ParseOptionalJsonNumber(jsonView, "SCALE_H_M", config.skySpectral.scaleHeightMie);
    ParseOptionalJsonFloat3(jsonView, "SUN_RADIANCE", config.skySpectral.sunRadiance);
    ParseOptionalJsonNumber(jsonView, "SUN_RADIUS", config.skySpectral.sunRadius);
    ParseOptionalJsonNumber(jsonView, "SUN_AA", config.skySpectral.sunAa);
    ParseOptionalJsonUint32(jsonView, "secondarySamples", config.skySpectral.secondarySamples);
    ParseOptionalJsonUint32(jsonView, "VIEW_STEPS", config.skySpectral.viewSteps);
    ParseOptionalJsonUint32(jsonView, "Samples", config.skySpectral.samples);
    ParseOptionalJsonVec3(jsonView, "position", config.sphere.position);
    ParseOptionalJsonVec3(jsonView, "rotationDegrees", config.sphere.rotationDegrees);
    ParseOptionalJsonVec3(jsonView, "scale", config.sphere.scale);
    ParseOptionalJsonUint32(jsonView, "longitudeSegments", config.sphere.longitudeSegments);
    ParseOptionalJsonUint32(jsonView, "latitudeSegments", config.sphere.latitudeSegments);
    ParseOptionalJsonFloat3(jsonView, "albedo", config.sphere.material.albedo);
    ParseOptionalJsonFloat3(jsonView, "emission", config.sphere.material.emission);
    ParseOptionalJsonNumber(jsonView, "roughness", config.sphere.material.roughness);
    ParseOptionalJsonFloat3(jsonView, "eta", config.sphere.material.eta);
    ParseOptionalJsonFloat3(jsonView, "extinction", config.sphere.material.extinction);

    if (config.width == 0 || config.height == 0)
    {
        throw std::runtime_error("\"width\" and \"height\" must be greater than 0.");
    }
    if (config.frameCount == 0)
    {
        throw std::runtime_error("\"frameCount\" must be greater than 0.");
    }
    if (config.samplesPerPixel == 0)
    {
        throw std::runtime_error("\"samplesPerPixel\" must be greater than 0.");
    }
    if (config.maxBounces < 2)
    {
        throw std::runtime_error("\"maxBounces\" must be at least 2.");
    }
    if (Length(config.initialLookAt - config.initialPosition) <= 0.001f)
    {
        throw std::runtime_error("\"initialPosition\" and \"initialLookAt\" must not be the same.");
    }
    if (config.fovYDegrees <= 1.0f || config.fovYDegrees >= 179.0f)
    {
        throw std::runtime_error("\"fovYDegrees\" must be between 1 and 179.");
    }
    if (config.moveSpeed < 0.0f || config.fastMoveSpeed < 0.0f)
    {
        throw std::runtime_error("\"moveSpeed\" and \"fastMoveSpeed\" must be non-negative.");
    }
    if (config.mouseSensitivity < 0.0f || config.keyLookSpeed < 0.0f)
    {
        throw std::runtime_error("\"mouseSensitivity\" and \"keyLookSpeed\" must be non-negative.");
    }
    if (config.maxPitchDegrees <= 0.0f || config.maxPitchDegrees >= 90.0f)
    {
        throw std::runtime_error("\"maxPitchDegrees\" must be greater than 0 and less than 90.");
    }
    if (config.skyExposure <= 0.0f)
    {
        throw std::runtime_error("\"exposure\" must be greater than 0.");
    }
    if (HasNegativeElement(config.skySpectral.betaRayleigh) || config.skySpectral.betaMie < 0.0f)
    {
        throw std::runtime_error("Sky scattering coefficients must be non-negative.");
    }
    if (config.skySpectral.mieG <= -1.0f || config.skySpectral.mieG >= 1.0f)
    {
        throw std::runtime_error("\"MIE_G\" must be between -1 and 1.");
    }
    if (config.skySpectral.earthRadius <= 0.0f || config.skySpectral.atmosphereRadius <= 0.0f
        || config.skySpectral.atmosphereRadius <= config.skySpectral.earthRadius)
    {
        throw std::runtime_error("\"EARTH_R\" and \"ATMOS_R\" must be positive, and ATMOS_R must exceed EARTH_R.");
    }
    if (config.skySpectral.scaleHeightRayleigh <= 0.0f || config.skySpectral.scaleHeightMie <= 0.0f)
    {
        throw std::runtime_error("\"SCALE_H_R\" and \"SCALE_H_M\" must be greater than 0.");
    }
    if (HasNegativeElement(config.skySpectral.sunRadiance))
    {
        throw std::runtime_error("\"SUN_RADIANCE\" values must be non-negative.");
    }
    if (config.skySpectral.sunRadius <= 0.0f || config.skySpectral.sunAa < 0.0f)
    {
        throw std::runtime_error("\"SUN_RADIUS\" must be greater than 0 and \"SUN_AA\" must be non-negative.");
    }
    if (config.skySpectral.secondarySamples == 0 || config.skySpectral.viewSteps == 0 || config.skySpectral.samples == 0)
    {
        throw std::runtime_error("Sky sample counts must be greater than 0.");
    }
    if (config.sphere.scale.x <= 0.0f || config.sphere.scale.y <= 0.0f || config.sphere.scale.z <= 0.0f)
    {
        throw std::runtime_error("\"scale\" must have positive values for all three axes.");
    }
    if (config.sphere.longitudeSegments < 3)
    {
        throw std::runtime_error("\"longitudeSegments\" must be at least 3.");
    }
    if (config.sphere.latitudeSegments < 2)
    {
        throw std::runtime_error("\"latitudeSegments\" must be at least 2.");
    }
    if (HasNegativeElement(config.sphere.material.albedo) || config.sphere.material.albedo[0] > 1.0f
        || config.sphere.material.albedo[1] > 1.0f || config.sphere.material.albedo[2] > 1.0f)
    {
        throw std::runtime_error("\"albedo\" values must be between 0 and 1.");
    }
    if (HasNegativeElement(config.sphere.material.emission))
    {
        throw std::runtime_error("\"emission\" values must be non-negative.");
    }
    if (config.sphere.material.roughness < 0.0f || config.sphere.material.roughness > 1.0f)
    {
        throw std::runtime_error("\"roughness\" must be between 0 and 1.");
    }
    if (HasNegativeElement(config.sphere.material.eta))
    {
        throw std::runtime_error("\"eta\" values must be non-negative.");
    }
    if (HasNegativeElement(config.sphere.material.extinction))
    {
        throw std::runtime_error("\"extinction\" values must be non-negative.");
    }

    return config;
}
