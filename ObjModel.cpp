#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "ObjModel.h"

#include <windows.h>

#include <array>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
struct ObjIndex
{
    int position = 0;
    int normal = 0;
};

Vec3 NormalizeSafe(const Vec3& value)
{
    const float length = Length(value);
    if (length <= 1.0e-6f)
    {
        return {0.0f, 1.0f, 0.0f};
    }
    return value * (1.0f / length);
}

int ResolveObjIndex(int index, size_t count, const std::string& context)
{
    if (index > 0)
    {
        const int resolved = index - 1;
        if (resolved >= static_cast<int>(count))
        {
            throw std::runtime_error(context + " references an index outside the source data.");
        }
        return resolved;
    }

    if (index < 0)
    {
        const int resolved = static_cast<int>(count) + index;
        if (resolved < 0 || resolved >= static_cast<int>(count))
        {
            throw std::runtime_error(context + " references an index outside the source data.");
        }
        return resolved;
    }

    throw std::runtime_error(context + " uses OBJ index 0, which is invalid.");
}

int ParseObjInteger(std::string_view token, const std::string& context)
{
    int value = 0;
    const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (ec != std::errc{} || ptr != token.data() + token.size())
    {
        throw std::runtime_error(context + " contains an invalid integer index.");
    }
    return value;
}

ObjIndex ParseFaceIndex(std::string_view token, const std::string& context)
{
    ObjIndex result{};

    const size_t firstSlash = token.find('/');
    if (firstSlash == std::string_view::npos)
    {
        result.position = ParseObjInteger(token, context);
        return result;
    }

    result.position = ParseObjInteger(token.substr(0, firstSlash), context);

    const size_t secondSlash = token.find('/', firstSlash + 1);
    if (secondSlash == std::string_view::npos)
    {
        return result;
    }

    const std::string_view normalToken = token.substr(secondSlash + 1);
    if (!normalToken.empty())
    {
        result.normal = ParseObjInteger(normalToken, context);
    }

    return result;
}

ModelVertex MakeModelVertex(const Vec3& position, const Vec3& normal)
{
    ModelVertex vertex{};
    vertex.position[0] = position.x;
    vertex.position[1] = position.y;
    vertex.position[2] = position.z;
    vertex.position[3] = 1.0f;
    vertex.normal[0] = normal.x;
    vertex.normal[1] = normal.y;
    vertex.normal[2] = normal.z;
    return vertex;
}
} // namespace

std::filesystem::path ResolveModelFilePath(const std::string& fileName)
{
    const std::filesystem::path filePath(fileName);
    if (filePath.is_absolute() && std::filesystem::exists(filePath))
    {
        return filePath;
    }

    WCHAR exePath[MAX_PATH]{};
    const DWORD pathLen = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    if (pathLen == 0 || pathLen == MAX_PATH)
    {
        throw std::runtime_error("Failed to resolve executable path.");
    }

    const auto exeDir = std::filesystem::path(exePath).parent_path();
    const std::array<std::filesystem::path, 5> candidatePaths = {
        exeDir / "Models" / filePath,
        exeDir.parent_path() / "Models" / filePath,
        std::filesystem::current_path() / "Models" / filePath,
        exeDir / filePath,
        std::filesystem::current_path() / filePath,
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

ObjModel LoadObjModel(const std::filesystem::path& filePath)
{
    std::ifstream file(filePath);
    if (!file)
    {
        throw std::runtime_error("Failed to open OBJ file: " + filePath.string());
    }

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    ObjModel model{};

    std::string line;
    uint32_t lineNumber = 0;
    while (std::getline(file, line))
    {
        ++lineNumber;
        std::stringstream lineStream(line);
        std::string keyword;
        lineStream >> keyword;
        if (keyword.empty() || keyword[0] == '#')
        {
            continue;
        }

        if (keyword == "v")
        {
            Vec3 position{};
            if (!(lineStream >> position.x >> position.y >> position.z))
            {
                throw std::runtime_error("Invalid vertex position in OBJ file at line " + std::to_string(lineNumber) + ".");
            }
            positions.push_back(position);
            continue;
        }

        if (keyword == "vn")
        {
            Vec3 normal{};
            if (!(lineStream >> normal.x >> normal.y >> normal.z))
            {
                throw std::runtime_error("Invalid vertex normal in OBJ file at line " + std::to_string(lineNumber) + ".");
            }
            normals.push_back(NormalizeSafe(normal));
            continue;
        }

        if (keyword != "f")
        {
            continue;
        }

        std::vector<ObjIndex> faceIndices;
        std::string token;
        while (lineStream >> token)
        {
            faceIndices.push_back(ParseFaceIndex(token, "Face at line " + std::to_string(lineNumber)));
        }

        if (faceIndices.size() < 3)
        {
            throw std::runtime_error("OBJ faces must contain at least three vertices.");
        }

        auto emitVertex = [&](const ObjIndex& index, const Vec3& fallbackNormal)
        {
            const int positionIndex =
                ResolveObjIndex(index.position, positions.size(), "Face at line " + std::to_string(lineNumber));
            const Vec3 position = positions[static_cast<size_t>(positionIndex)];

            Vec3 normal = fallbackNormal;
            if (index.normal != 0)
            {
                const int normalIndex =
                    ResolveObjIndex(index.normal, normals.size(), "Face at line " + std::to_string(lineNumber));
                normal = normals[static_cast<size_t>(normalIndex)];
            }

            const uint32_t vertexIndex = static_cast<uint32_t>(model.vertices.size());
            model.vertices.push_back(MakeModelVertex(position, normal));
            model.indices.push_back(vertexIndex);
        };

        for (size_t i = 1; i + 1 < faceIndices.size(); ++i)
        {
            const int i0 = ResolveObjIndex(faceIndices[0].position,
                                           positions.size(),
                                           "Face at line " + std::to_string(lineNumber));
            const int i1 = ResolveObjIndex(faceIndices[i].position,
                                           positions.size(),
                                           "Face at line " + std::to_string(lineNumber));
            const int i2 = ResolveObjIndex(faceIndices[i + 1].position,
                                           positions.size(),
                                           "Face at line " + std::to_string(lineNumber));

            const Vec3 p0 = positions[static_cast<size_t>(i0)];
            const Vec3 p1 = positions[static_cast<size_t>(i1)];
            const Vec3 p2 = positions[static_cast<size_t>(i2)];
            const Vec3 faceNormal = NormalizeSafe(Cross(p1 - p0, p2 - p0));

            emitVertex(faceIndices[0], faceNormal);
            emitVertex(faceIndices[i], faceNormal);
            emitVertex(faceIndices[i + 1], faceNormal);
        }
    }

    if (model.vertices.empty() || model.indices.empty())
    {
        throw std::runtime_error("OBJ file does not contain any triangle geometry: " + filePath.string());
    }

    return model;
}
