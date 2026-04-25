#pragma once

#include "RuntimeConfig.h"

#include <cstdint>
#include <filesystem>
#include <vector>

struct ModelVertex
{
    float position[4];
    float normal[4];
};

struct ObjModel
{
    std::vector<ModelVertex> vertices;
    std::vector<uint32_t> indices;
};

std::filesystem::path ResolveModelFilePath(const std::string& fileName);
ObjModel LoadObjModel(const std::filesystem::path& filePath);
