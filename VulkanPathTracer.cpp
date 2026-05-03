#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#define VK_USE_PLATFORM_WIN32_KHR

#include "VulkanPathTracer.h"

#include "ObjModel.h"
#include "RuntimeConfig.h"

#include <windows.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <vector>

// SECTION: helpers

static void ThrowIfFalse(BOOL condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

static void ThrowVk(VkResult result, const char* message)
{
    if (result != VK_SUCCESS)
    {
        char buffer[256]{};
        std::snprintf(buffer, sizeof(buffer), "%s (VkResult=%d)", message, static_cast<int>(result));
        throw std::runtime_error(buffer);
    }
}

static std::vector<char> LoadBinaryFile(const wchar_t* fileName)
{
    const auto shaderPath = ResolveRuntimeFilePath(fileName);
    if (shaderPath.empty())
    {
        throw std::runtime_error("Failed to open SPIR-V shader.");
    }

    std::ifstream file(shaderPath, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Failed to open SPIR-V shader.");
    }
    const auto fileSize = file.tellg();
    if (fileSize <= 0)
    {
        throw std::runtime_error("SPIR-V shader is empty.");
    }

    std::vector<char> data(static_cast<size_t>(fileSize));
    file.seekg(0, std::ios::beg);
    file.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!file)
    {
        throw std::runtime_error("Failed to read SPIR-V shader.");
    }

    return data;
}

static VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment)
{
    if (alignment == 0)
    {
        return value;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool IsComplete() const
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapchainSupport
{
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct BufferAllocation
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct AccelerationStructureAllocation
{
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    BufferAllocation buffer;
};

struct alignas(16) SceneData
{
    float skyBetaRayleighBetaM[4];
    float skyMieEarthAtmosScaleHr[4];
    float skyScaleHmSunRadiusAa[4];
    float skySunRadiance[4];
    uint32_t skySampleCounts[4];
};

static_assert(sizeof(SceneData) == 80, "Scene data layout must stay 16-byte aligned.");

static_assert(sizeof(ModelVertex) == 32, "Model vertex layout must stay 16-byte aligned.");

struct alignas(16) InstanceData
{
    uint32_t materialIndexFirstIndex[4];
};

static_assert(sizeof(InstanceData) == 16, "Instance data layout must stay 16-byte aligned.");

struct alignas(16) MaterialData
{
    float albedoRoughness[4];
    float emission[4];
    float eta[4];
    float extinction[4];
};

static_assert(sizeof(MaterialData) == 64, "Material data layout must stay 16-byte aligned.");

struct ModelGeometryRange
{
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
};

struct ModelAccelerationStructure
{
    ModelGeometryRange geometry;
    AccelerationStructureAllocation blas;
};

struct Mat3
{
    float m[3][3]{};
};

static Mat3 IdentityMat3()
{
    Mat3 result{};
    result.m[0][0] = 1.0f;
    result.m[1][1] = 1.0f;
    result.m[2][2] = 1.0f;
    return result;
}

static Mat3 Multiply(const Mat3& left, const Mat3& right)
{
    Mat3 result{};
    for (uint32_t row = 0; row < 3; ++row)
    {
        for (uint32_t column = 0; column < 3; ++column)
        {
            result.m[row][column] = left.m[row][0] * right.m[0][column]
                                    + left.m[row][1] * right.m[1][column]
                                    + left.m[row][2] * right.m[2][column];
        }
    }
    return result;
}

static Mat3 Transpose(const Mat3& matrix)
{
    Mat3 result{};
    for (uint32_t row = 0; row < 3; ++row)
    {
        for (uint32_t column = 0; column < 3; ++column)
        {
            result.m[row][column] = matrix.m[column][row];
        }
    }
    return result;
}

static Vec3 TransformDirection(const Mat3& matrix, const Vec3& value)
{
    return {
        matrix.m[0][0] * value.x + matrix.m[0][1] * value.y + matrix.m[0][2] * value.z,
        matrix.m[1][0] * value.x + matrix.m[1][1] * value.y + matrix.m[1][2] * value.z,
        matrix.m[2][0] * value.x + matrix.m[2][1] * value.y + matrix.m[2][2] * value.z,
    };
}

static Mat3 MakeScaleMatrix(const Vec3& scale)
{
    Mat3 result{};
    result.m[0][0] = scale.x;
    result.m[1][1] = scale.y;
    result.m[2][2] = scale.z;
    return result;
}

static Mat3 MakeRotationX(float radians)
{
    Mat3 result = IdentityMat3();
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);
    result.m[1][1] = cosine;
    result.m[1][2] = -sine;
    result.m[2][1] = sine;
    result.m[2][2] = cosine;
    return result;
}

static Mat3 MakeRotationY(float radians)
{
    Mat3 result = IdentityMat3();
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);
    result.m[0][0] = cosine;
    result.m[0][2] = sine;
    result.m[2][0] = -sine;
    result.m[2][2] = cosine;
    return result;
}

static Mat3 MakeRotationZ(float radians)
{
    Mat3 result = IdentityMat3();
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);
    result.m[0][0] = cosine;
    result.m[0][1] = -sine;
    result.m[1][0] = sine;
    result.m[1][1] = cosine;
    return result;
}

static Mat3 MakeEulerRotationMatrixDegrees(const Vec3& rotationDegrees)
{
    const float radiansX = rotationDegrees.x * kPi / 180.0f;
    const float radiansY = rotationDegrees.y * kPi / 180.0f;
    const float radiansZ = rotationDegrees.z * kPi / 180.0f;
    return Multiply(Multiply(MakeRotationZ(radiansZ), MakeRotationY(radiansY)), MakeRotationX(radiansX));
}

static bool HasSkySpectralChanged(const SkySpectralConfig& left, const SkySpectralConfig& right)
{
    return left.betaRayleigh != right.betaRayleigh
           || left.betaMie != right.betaMie
           || left.mieG != right.mieG
           || left.earthRadius != right.earthRadius
           || left.atmosphereRadius != right.atmosphereRadius
           || left.scaleHeightRayleigh != right.scaleHeightRayleigh
           || left.scaleHeightMie != right.scaleHeightMie
           || left.sunRadiance != right.sunRadiance
           || left.sunRadius != right.sunRadius
           || left.sunAa != right.sunAa
           || left.secondarySamples != right.secondarySamples
           || left.viewSteps != right.viewSteps
           || left.samples != right.samples;
}

struct PushConstants
{
    float cameraPositionFrame[4];
    float cameraForwardSamples[4];
    float cameraRightBounces[4];
    float cameraUpTanHalfFovY[4];
    float skyBottomExposure[4];
    float skyTopAspect[4];
    uint32_t imageSize[2];
};

static_assert(sizeof(PushConstants) == 104, "Push constant layout must match the shader.");

// SECTION: window procedure

static LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(window, message, wParam, lParam);
    }
}

// SECTION: Vulkan app

class VulkanPathTracer
{
public:
    static constexpr const wchar_t* CONFIG_FILE_NAME = L"path_tracer_config.json";

    ~VulkanPathTracer()
    {
        Cleanup();
    }

    void Run()
    {
        LoadInitialRuntimeConfig();
        CreateWindowAndShow();
        CreateInstance();
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        // Command pool is created before ray tracing scene setup because AS builds use one-time command buffers.
        CreateCommandPool();
        CreateRayTracingScene();
        CreateSwapchain();
        CreateDescriptorSetLayout();
        CreatePipeline();
        CreateShaderBindingTables();
        CreateDescriptorSets();
        CreateCommandBuffers();
        CreateSyncObjects();
        MessageLoop();
        vkDeviceWaitIdle(m_device);
    }

private:
    static bool IsKeyDown(int virtualKey)
    {
        return (GetAsyncKeyState(virtualKey) & 0x8000) != 0;
    }

    static bool HasModelDefinitionsChanged(const RuntimeConfig& left, const RuntimeConfig& right)
    {
        return left.models != right.models;
    }

    static bool HasInstanceTransformChanged(const RuntimeConfig& left, const RuntimeConfig& right)
    {
        if (left.instances.size() != right.instances.size())
        {
            return true;
        }

        for (size_t i = 0; i < left.instances.size(); ++i)
        {
            if (left.instances[i].position != right.instances[i].position
                || left.instances[i].rotationDegrees != right.instances[i].rotationDegrees
                || left.instances[i].scale != right.instances[i].scale)
            {
                return true;
            }
        }

        return false;
    }

    static bool HasInstanceBindingChanged(const RuntimeConfig& left, const RuntimeConfig& right)
    {
        if (left.instances.size() != right.instances.size())
        {
            return true;
        }

        for (size_t i = 0; i < left.instances.size(); ++i)
        {
            if (left.instances[i].modelIndex != right.instances[i].modelIndex
                || left.instances[i].materialIndex != right.instances[i].materialIndex)
            {
                return true;
            }
        }

        return false;
    }

    static bool HasMaterialDataChanged(const RuntimeConfig& left, const RuntimeConfig& right)
    {
        return left.materials != right.materials;
    }

    float GetCameraMaxPitchRadians() const
    {
        return m_config.maxPitchDegrees * kPi / 180.0f;
    }

    SceneData BuildSceneData() const
    {
        SceneData sceneData{};
        sceneData.skyBetaRayleighBetaM[0] = m_config.skySpectral.betaRayleigh[0];
        sceneData.skyBetaRayleighBetaM[1] = m_config.skySpectral.betaRayleigh[1];
        sceneData.skyBetaRayleighBetaM[2] = m_config.skySpectral.betaRayleigh[2];
        sceneData.skyBetaRayleighBetaM[3] = m_config.skySpectral.betaMie;
        sceneData.skyMieEarthAtmosScaleHr[0] = m_config.skySpectral.mieG;
        sceneData.skyMieEarthAtmosScaleHr[1] = m_config.skySpectral.earthRadius;
        sceneData.skyMieEarthAtmosScaleHr[2] = m_config.skySpectral.atmosphereRadius;
        sceneData.skyMieEarthAtmosScaleHr[3] = m_config.skySpectral.scaleHeightRayleigh;
        sceneData.skyScaleHmSunRadiusAa[0] = m_config.skySpectral.scaleHeightMie;
        sceneData.skyScaleHmSunRadiusAa[1] = m_config.skySpectral.sunRadius;
        sceneData.skyScaleHmSunRadiusAa[2] = m_config.skySpectral.sunAa;
        sceneData.skySunRadiance[0] = m_config.skySpectral.sunRadiance[0];
        sceneData.skySunRadiance[1] = m_config.skySpectral.sunRadiance[1];
        sceneData.skySunRadiance[2] = m_config.skySpectral.sunRadiance[2];
        sceneData.skySampleCounts[0] = m_config.skySpectral.secondarySamples;
        sceneData.skySampleCounts[1] = m_config.skySpectral.viewSteps;
        sceneData.skySampleCounts[2] = m_config.skySpectral.samples;
        return sceneData;
    }

    InstanceData BuildInstanceData(const ModelInstanceConfig& instanceConfig, const ModelGeometryRange& geometry) const
    {
        InstanceData instanceData{};
        instanceData.materialIndexFirstIndex[0] = instanceConfig.materialIndex;
        instanceData.materialIndexFirstIndex[1] = geometry.firstIndex;
        return instanceData;
    }

    MaterialData BuildMaterialData(const MaterialConfig& materialConfig) const
    {
        MaterialData materialData{};
        materialData.albedoRoughness[0] = materialConfig.albedo[0];
        materialData.albedoRoughness[1] = materialConfig.albedo[1];
        materialData.albedoRoughness[2] = materialConfig.albedo[2];
        materialData.albedoRoughness[3] = materialConfig.roughness;
        materialData.emission[0] = materialConfig.emission[0];
        materialData.emission[1] = materialConfig.emission[1];
        materialData.emission[2] = materialConfig.emission[2];
        materialData.eta[0] = materialConfig.eta[0];
        materialData.eta[1] = materialConfig.eta[1];
        materialData.eta[2] = materialConfig.eta[2];
        materialData.extinction[0] = materialConfig.extinction[0];
        materialData.extinction[1] = materialConfig.extinction[1];
        materialData.extinction[2] = materialConfig.extinction[2];
        return materialData;
    }

    std::vector<InstanceData> BuildInstanceDataArray() const
    {
        std::vector<InstanceData> instanceData;
        instanceData.reserve(m_config.instances.size());
        for (const ModelInstanceConfig& instanceConfig : m_config.instances)
        {
            instanceData.push_back(
                BuildInstanceData(instanceConfig, m_modelAccelerationStructures[instanceConfig.modelIndex].geometry));
        }
        return instanceData;
    }

    std::vector<MaterialData> BuildMaterialDataArray() const
    {
        std::vector<MaterialData> materialData;
        materialData.reserve(m_config.materials.size());
        for (const MaterialConfig& materialConfig : m_config.materials)
        {
            materialData.push_back(BuildMaterialData(materialConfig));
        }
        return materialData;
    }

    std::vector<VkAccelerationStructureInstanceKHR> BuildSceneInstances() const
    {
        std::vector<VkAccelerationStructureInstanceKHR> instances;
        instances.reserve(m_config.instances.size());

        for (size_t instanceIndex = 0; instanceIndex < m_config.instances.size(); ++instanceIndex)
        {
            const ModelInstanceConfig& instanceConfig = m_config.instances[instanceIndex];
            const Mat3 rotation = MakeEulerRotationMatrixDegrees(instanceConfig.rotationDegrees);
            const Mat3 objectToWorldLinear = Multiply(rotation, MakeScaleMatrix(instanceConfig.scale));

            VkAccelerationStructureInstanceKHR instance{};
            instance.transform.matrix[0][0] = objectToWorldLinear.m[0][0];
            instance.transform.matrix[0][1] = objectToWorldLinear.m[0][1];
            instance.transform.matrix[0][2] = objectToWorldLinear.m[0][2];
            instance.transform.matrix[0][3] = instanceConfig.position.x;
            instance.transform.matrix[1][0] = objectToWorldLinear.m[1][0];
            instance.transform.matrix[1][1] = objectToWorldLinear.m[1][1];
            instance.transform.matrix[1][2] = objectToWorldLinear.m[1][2];
            instance.transform.matrix[1][3] = instanceConfig.position.y;
            instance.transform.matrix[2][0] = objectToWorldLinear.m[2][0];
            instance.transform.matrix[2][1] = objectToWorldLinear.m[2][1];
            instance.transform.matrix[2][2] = objectToWorldLinear.m[2][2];
            instance.transform.matrix[2][3] = instanceConfig.position.z;
            instance.instanceCustomIndex = static_cast<uint32_t>(instanceIndex);
            instance.mask = 0xFF;
            instance.instanceShaderBindingTableRecordOffset = 0;
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR;
            instance.accelerationStructureReference =
                GetAccelerationStructureDeviceAddress(m_modelAccelerationStructures[instanceConfig.modelIndex].blas.handle);
            instances.push_back(instance);
        }

        return instances;
    }

    void CreateSceneBuffers()
    {
        m_sceneDataBuffer = CreateBuffer(sizeof(SceneData),
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         false);
        m_instanceDataBuffer = CreateBuffer(static_cast<VkDeviceSize>(m_config.instances.size() * sizeof(InstanceData)),
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                            false);
        m_materialDataBuffer =
            CreateBuffer(static_cast<VkDeviceSize>(m_config.materials.size() * sizeof(MaterialData)),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         false);
    }

    void DestroySceneBuffers()
    {
        DestroyBuffer(m_sceneDataBuffer);
        DestroyBuffer(m_instanceDataBuffer);
        DestroyBuffer(m_materialDataBuffer);
    }

    void UploadSceneDataFromConfig()
    {
        const SceneData sceneData = BuildSceneData();
        UploadToBuffer(m_sceneDataBuffer, &sceneData, sizeof(sceneData));

        const std::vector<InstanceData> instanceData = BuildInstanceDataArray();
        UploadToBuffer(m_instanceDataBuffer, instanceData.data(), instanceData.size() * sizeof(InstanceData));

        const std::vector<MaterialData> materialData = BuildMaterialDataArray();
        UploadToBuffer(m_materialDataBuffer, materialData.data(), materialData.size() * sizeof(MaterialData));
    }

    void UpdateDescriptorSetContents()
    {
        if (m_descriptorSets.empty())
        {
            return;
        }

        for (size_t i = 0; i < m_descriptorSets.size(); ++i)
        {
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageView = m_swapchainImageViews[i];
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = m_descriptorSets[i];
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.pImageInfo = &imageInfo;

            VkWriteDescriptorSetAccelerationStructureKHR accelerationWriteInfo{};
            accelerationWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            accelerationWriteInfo.accelerationStructureCount = 1;
            accelerationWriteInfo.pAccelerationStructures = &m_tlas.handle;

            VkWriteDescriptorSet accelerationWrite{};
            accelerationWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            accelerationWrite.pNext = &accelerationWriteInfo;
            accelerationWrite.dstSet = m_descriptorSets[i];
            accelerationWrite.dstBinding = 1;
            accelerationWrite.descriptorCount = 1;
            accelerationWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

            VkDescriptorBufferInfo sceneDataInfo{};
            sceneDataInfo.buffer = m_sceneDataBuffer.buffer;
            sceneDataInfo.offset = 0;
            sceneDataInfo.range = m_sceneDataBuffer.size;

            VkWriteDescriptorSet sceneWrite{};
            sceneWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            sceneWrite.dstSet = m_descriptorSets[i];
            sceneWrite.dstBinding = 2;
            sceneWrite.descriptorCount = 1;
            sceneWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            sceneWrite.pBufferInfo = &sceneDataInfo;

            VkDescriptorBufferInfo instanceDataInfo{};
            instanceDataInfo.buffer = m_instanceDataBuffer.buffer;
            instanceDataInfo.offset = 0;
            instanceDataInfo.range = m_instanceDataBuffer.size;

            VkWriteDescriptorSet instanceWrite{};
            instanceWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            instanceWrite.dstSet = m_descriptorSets[i];
            instanceWrite.dstBinding = 3;
            instanceWrite.descriptorCount = 1;
            instanceWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            instanceWrite.pBufferInfo = &instanceDataInfo;

            VkDescriptorBufferInfo materialDataInfo{};
            materialDataInfo.buffer = m_materialDataBuffer.buffer;
            materialDataInfo.offset = 0;
            materialDataInfo.range = m_materialDataBuffer.size;

            VkWriteDescriptorSet materialWrite{};
            materialWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            materialWrite.dstSet = m_descriptorSets[i];
            materialWrite.dstBinding = 4;
            materialWrite.descriptorCount = 1;
            materialWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            materialWrite.pBufferInfo = &materialDataInfo;

            VkDescriptorBufferInfo vertexDataInfo{};
            vertexDataInfo.buffer = m_vertexBuffer.buffer;
            vertexDataInfo.offset = 0;
            vertexDataInfo.range = m_vertexBuffer.size;

            VkWriteDescriptorSet vertexWrite{};
            vertexWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            vertexWrite.dstSet = m_descriptorSets[i];
            vertexWrite.dstBinding = 5;
            vertexWrite.descriptorCount = 1;
            vertexWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            vertexWrite.pBufferInfo = &vertexDataInfo;

            VkDescriptorBufferInfo indexDataInfo{};
            indexDataInfo.buffer = m_indexBuffer.buffer;
            indexDataInfo.offset = 0;
            indexDataInfo.range = m_indexBuffer.size;

            VkWriteDescriptorSet indexWrite{};
            indexWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            indexWrite.dstSet = m_descriptorSets[i];
            indexWrite.dstBinding = 6;
            indexWrite.descriptorCount = 1;
            indexWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            indexWrite.pBufferInfo = &indexDataInfo;

            const std::array<VkWriteDescriptorSet, 7> writes = {
                write,
                accelerationWrite,
                sceneWrite,
                instanceWrite,
                materialWrite,
                vertexWrite,
                indexWrite,
            };
            vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }

    void DestroyGeometryResources()
    {
        if (m_tlas.handle != VK_NULL_HANDLE)
        {
            m_vkDestroyAccelerationStructureKHR(m_device, m_tlas.handle, nullptr);
            m_tlas.handle = VK_NULL_HANDLE;
        }
        DestroyBuffer(m_tlas.buffer);
        DestroyBuffer(m_instanceBuffer);

        for (auto& model : m_modelAccelerationStructures)
        {
            if (model.blas.handle != VK_NULL_HANDLE)
            {
                m_vkDestroyAccelerationStructureKHR(m_device, model.blas.handle, nullptr);
                model.blas.handle = VK_NULL_HANDLE;
            }
            DestroyBuffer(model.blas.buffer);
        }
        m_modelAccelerationStructures.clear();
        DestroyBuffer(m_indexBuffer);
        DestroyBuffer(m_vertexBuffer);
    }

    void CreateGeometryResourcesFromConfig()
    {
        std::vector<ModelVertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<ModelGeometryRange> geometryRanges;
        geometryRanges.reserve(m_config.models.size());

        for (const ModelAssetConfig& modelConfig : m_config.models)
        {
            const std::filesystem::path modelPath = ResolveModelFilePath(modelConfig.fileName);
            if (modelPath.empty())
            {
                throw std::runtime_error("Failed to locate OBJ model: " + modelConfig.fileName);
            }

            const ObjModel loadedModel = LoadObjModel(modelPath);
            ModelGeometryRange geometry{};
            geometry.firstVertex = static_cast<uint32_t>(vertices.size());
            geometry.vertexCount = static_cast<uint32_t>(loadedModel.vertices.size());
            geometry.firstIndex = static_cast<uint32_t>(indices.size());
            geometry.indexCount = static_cast<uint32_t>(loadedModel.indices.size());

            vertices.insert(vertices.end(), loadedModel.vertices.begin(), loadedModel.vertices.end());
            for (uint32_t index : loadedModel.indices)
            {
                indices.push_back(index + geometry.firstVertex);
            }

            geometryRanges.push_back(geometry);
        }

        m_vertexBuffer = CreateBuffer(static_cast<VkDeviceSize>(vertices.size() * sizeof(ModelVertex)),
                                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                          | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      true);
        m_indexBuffer = CreateBuffer(static_cast<VkDeviceSize>(indices.size() * sizeof(uint32_t)),
                                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                         | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     true);
        UploadToBuffer(m_vertexBuffer, vertices.data(), vertices.size() * sizeof(ModelVertex));
        UploadToBuffer(m_indexBuffer, indices.data(), indices.size() * sizeof(uint32_t));

        const VkDeviceAddress vertexAddress = GetBufferDeviceAddress(m_vertexBuffer.buffer);
        const VkDeviceAddress indexAddress = GetBufferDeviceAddress(m_indexBuffer.buffer);

        m_modelAccelerationStructures.clear();
        m_modelAccelerationStructures.reserve(geometryRanges.size());
        for (const ModelGeometryRange& geometry : geometryRanges)
        {
            VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
            triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triangles.vertexData.deviceAddress = vertexAddress;
            triangles.vertexStride = sizeof(ModelVertex);
            triangles.maxVertex = geometry.firstVertex + geometry.vertexCount - 1;
            triangles.indexType = VK_INDEX_TYPE_UINT32;
            triangles.indexData.deviceAddress =
                indexAddress + static_cast<VkDeviceSize>(geometry.firstIndex) * sizeof(uint32_t);

            VkAccelerationStructureGeometryKHR blasGeometry{};
            blasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            blasGeometry.geometry.triangles = triangles;

            VkAccelerationStructureBuildGeometryInfoKHR blasBuildInfo{};
            blasBuildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            blasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            blasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            blasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            blasBuildInfo.geometryCount = 1;
            blasBuildInfo.pGeometries = &blasGeometry;

            uint32_t blasPrimitiveCount = geometry.indexCount / 3;
            VkAccelerationStructureBuildSizesInfoKHR blasSizeInfo{};
            blasSizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            m_vkGetAccelerationStructureBuildSizesKHR(m_device,
                                                      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                      &blasBuildInfo,
                                                      &blasPrimitiveCount,
                                                      &blasSizeInfo);

            ModelAccelerationStructure modelResources{};
            modelResources.geometry = geometry;
            modelResources.blas = CreateAccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                                              blasSizeInfo.accelerationStructureSize);
            blasBuildInfo.dstAccelerationStructure = modelResources.blas.handle;

            BufferAllocation blasScratch = CreateBuffer(blasSizeInfo.buildScratchSize,
                                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                        true);
            blasBuildInfo.scratchData.deviceAddress = GetBufferDeviceAddress(blasScratch.buffer);

            VkAccelerationStructureBuildRangeInfoKHR blasRangeInfo{};
            blasRangeInfo.primitiveCount = blasPrimitiveCount;
            const VkAccelerationStructureBuildRangeInfoKHR* blasRangeInfos[] = {&blasRangeInfo};

            VkCommandBuffer blasCommandBuffer = BeginSingleTimeCommands();
            m_vkCmdBuildAccelerationStructuresKHR(blasCommandBuffer, 1, &blasBuildInfo, blasRangeInfos);
            EndSingleTimeCommands(blasCommandBuffer);
            DestroyBuffer(blasScratch);

            m_modelAccelerationStructures.push_back(modelResources);
        }
    }

    void RebuildTopLevelAccelerationStructure()
    {
        if (m_tlas.handle != VK_NULL_HANDLE)
        {
            m_vkDestroyAccelerationStructureKHR(m_device, m_tlas.handle, nullptr);
            m_tlas.handle = VK_NULL_HANDLE;
        }
        DestroyBuffer(m_tlas.buffer);
        DestroyBuffer(m_instanceBuffer);

        const std::vector<VkAccelerationStructureInstanceKHR> instances = BuildSceneInstances();
        m_instanceBuffer = CreateBuffer(static_cast<VkDeviceSize>(instances.size() * sizeof(VkAccelerationStructureInstanceKHR)),
                                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                        true);
        UploadToBuffer(m_instanceBuffer,
                       instances.data(),
                       instances.size() * sizeof(VkAccelerationStructureInstanceKHR));

        VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
        instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        instancesData.data.deviceAddress = GetBufferDeviceAddress(m_instanceBuffer.buffer);

        VkAccelerationStructureGeometryKHR tlasGeometry{};
        tlasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        tlasGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlasGeometry.geometry.instances = instancesData;

        VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{};
        tlasBuildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        tlasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tlasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        tlasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        tlasBuildInfo.geometryCount = 1;
        tlasBuildInfo.pGeometries = &tlasGeometry;

        uint32_t tlasPrimitiveCount = static_cast<uint32_t>(instances.size());
        VkAccelerationStructureBuildSizesInfoKHR tlasSizeInfo{};
        tlasSizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        m_vkGetAccelerationStructureBuildSizesKHR(m_device,
                                                  VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                  &tlasBuildInfo,
                                                  &tlasPrimitiveCount,
                                                  &tlasSizeInfo);

        m_tlas = CreateAccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                             tlasSizeInfo.accelerationStructureSize);
        tlasBuildInfo.dstAccelerationStructure = m_tlas.handle;

        BufferAllocation tlasScratch = CreateBuffer(tlasSizeInfo.buildScratchSize,
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                    true);
        tlasBuildInfo.scratchData.deviceAddress = GetBufferDeviceAddress(tlasScratch.buffer);

        VkAccelerationStructureBuildRangeInfoKHR tlasRangeInfo{};
        tlasRangeInfo.primitiveCount = tlasPrimitiveCount;
        const VkAccelerationStructureBuildRangeInfoKHR* tlasRangeInfos[] = {&tlasRangeInfo};

        VkCommandBuffer tlasCommandBuffer = BeginSingleTimeCommands();
        m_vkCmdBuildAccelerationStructuresKHR(tlasCommandBuffer, 1, &tlasBuildInfo, tlasRangeInfos);
        EndSingleTimeCommands(tlasCommandBuffer);
        DestroyBuffer(tlasScratch);
    }

    void RefreshSceneFromConfig(bool rebuildGeometryResources,
                                bool rebuildTopLevelAccelerationStructure,
                                bool recreateSceneBuffers)
    {
        if (m_sceneDataBuffer.buffer == VK_NULL_HANDLE)
        {
            return;
        }

        ThrowVk(vkDeviceWaitIdle(m_device), "Failed to wait for device idle during scene reload");
        if (rebuildGeometryResources)
        {
            DestroyGeometryResources();
            CreateGeometryResourcesFromConfig();
        }
        if (recreateSceneBuffers)
        {
            DestroySceneBuffers();
            CreateSceneBuffers();
        }
        UploadSceneDataFromConfig();

        bool descriptorSetsNeedUpdate = recreateSceneBuffers || rebuildGeometryResources;
        if (rebuildTopLevelAccelerationStructure)
        {
            RebuildTopLevelAccelerationStructure();
            descriptorSetsNeedUpdate = true;
        }
        if (descriptorSetsNeedUpdate)
        {
            UpdateDescriptorSetContents();
        }
    }

    void ApplyRuntimeConfig(const RuntimeConfig& config, bool resetCameraState)
    {
        const bool modelDefinitionsChanged = HasModelDefinitionsChanged(config, m_config);
        const bool instanceTransformChanged = HasInstanceTransformChanged(config, m_config);
        const bool instanceBindingChanged = HasInstanceBindingChanged(config, m_config);
        const bool materialDataChanged = HasMaterialDataChanged(config, m_config);
        const bool sceneBufferLayoutChanged =
            config.instances.size() != m_config.instances.size() || config.materials.size() != m_config.materials.size();
        const bool skySpectralChanged = HasSkySpectralChanged(config.skySpectral, m_config.skySpectral);
        if (m_physicalDevice != VK_NULL_HANDLE
            && config.maxBounces > m_rayTracingPipelineProperties.maxRayRecursionDepth)
        {
            throw std::runtime_error("\"maxBounces\" exceeds the selected GPU ray recursion depth.");
        }

        if (!m_configPath.empty() && !resetCameraState)
        {
            if (config.width != m_config.width || config.height != m_config.height
                || config.frameCount != m_config.frameCount)
            {
                std::puts("[Config] width/height/frameCount changes apply on the next launch.");
            }
        }

        m_config = config;
        if (resetCameraState)
        {
            ResetCamera();
            return;
        }

        const float maxPitch = GetCameraMaxPitchRadians();
        m_cameraPitch = std::clamp(m_cameraPitch, -maxPitch, maxPitch);

        if (modelDefinitionsChanged || instanceTransformChanged || instanceBindingChanged || materialDataChanged
            || skySpectralChanged)
        {
            RefreshSceneFromConfig(modelDefinitionsChanged,
                                   modelDefinitionsChanged || instanceTransformChanged || instanceBindingChanged,
                                   sceneBufferLayoutChanged);
        }
    }

    void LoadInitialRuntimeConfig()
    {
        m_configPath = ResolveRuntimeFilePath(CONFIG_FILE_NAME);
        if (m_configPath.empty())
        {
            throw std::runtime_error("Failed to locate path_tracer_config.json.");
        }

        ApplyRuntimeConfig(ParseRuntimeConfig(LoadTextFile(m_configPath)), true);

        std::error_code errorCode;
        m_configLastWriteTime = std::filesystem::last_write_time(m_configPath, errorCode);
        if (errorCode)
        {
            throw std::runtime_error("Failed to read config file timestamp.");
        }

        std::printf("[Config] Loaded %s\n", m_configPath.string().c_str());
    }

    void ReloadRuntimeConfigIfNeeded()
    {
        if (m_configPath.empty())
        {
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now - m_lastConfigPollTime < std::chrono::milliseconds(250))
        {
            return;
        }
        m_lastConfigPollTime = now;

        std::error_code errorCode;
        const auto currentWriteTime = std::filesystem::last_write_time(m_configPath, errorCode);
        if (errorCode || currentWriteTime == m_configLastWriteTime)
        {
            return;
        }

        try
        {
            ApplyRuntimeConfig(ParseRuntimeConfig(LoadTextFile(m_configPath)), false);
            std::printf("[Config] Reloaded %s\n", m_configPath.string().c_str());
        }
        catch (const std::exception& error)
        {
            std::fprintf(stderr, "[Config] Reload failed: %s\n", error.what());
        }

        m_configLastWriteTime = currentWriteTime;
    }

    void ResetCamera()
    {
        m_cameraPosition = m_config.initialPosition;
        const Vec3 initialForward = Normalize(m_config.initialLookAt - m_cameraPosition);
        m_cameraYaw = std::atan2(initialForward.x, initialForward.z);
        m_cameraPitch = std::asin(std::clamp(initialForward.y, -1.0f, 1.0f));
    }

    Vec3 GetCameraForward() const
    {
        const float cosPitch = std::cos(m_cameraPitch);
        return Normalize({
            std::sin(m_cameraYaw) * cosPitch,
            std::sin(m_cameraPitch),
            std::cos(m_cameraYaw) * cosPitch,
        });
    }

    Vec3 GetCameraPlanarForward() const
    {
        return {std::sin(m_cameraYaw), 0.0f, std::cos(m_cameraYaw)};
    }

    void UpdateMouseLook()
    {
        const bool windowFocused = GetForegroundWindow() == m_window;
        const bool wantsMouseLook = windowFocused && IsKeyDown(VK_RBUTTON);
        if (!wantsMouseLook)
        {
            if (m_mouseLookActive && GetCapture() == m_window)
            {
                ReleaseCapture();
            }
            m_mouseLookActive = false;
            return;
        }

        POINT cursorPosition{};
        if (!GetCursorPos(&cursorPosition))
        {
            return;
        }

        if (!m_mouseLookActive)
        {
            m_mouseLookActive = true;
            m_lastMousePosition = cursorPosition;
            SetCapture(m_window);
            return;
        }

        const float mouseDeltaX = static_cast<float>(cursorPosition.x - m_lastMousePosition.x);
        const float mouseDeltaY = static_cast<float>(cursorPosition.y - m_lastMousePosition.y);
        m_lastMousePosition = cursorPosition;

        const float maxPitch = GetCameraMaxPitchRadians();
        m_cameraYaw += mouseDeltaX * m_config.mouseSensitivity;
        m_cameraPitch = std::clamp(m_cameraPitch - mouseDeltaY * m_config.mouseSensitivity, -maxPitch, maxPitch);
    }

    void UpdateCamera(double deltaSeconds)
    {
        const float deltaTime = static_cast<float>(std::min(deltaSeconds, 0.1));
        UpdateMouseLook();

        const bool windowFocused = GetForegroundWindow() == m_window;
        const bool resetCameraDown = windowFocused && IsKeyDown('R');
        if (resetCameraDown && !m_resetCameraKeyDown)
        {
            ResetCamera();
        }
        m_resetCameraKeyDown = resetCameraDown;

        if (!windowFocused)
        {
            return;
        }

        const float maxPitch = GetCameraMaxPitchRadians();
        if (IsKeyDown(VK_LEFT))
        {
            m_cameraYaw -= m_config.keyLookSpeed * deltaTime;
        }
        if (IsKeyDown(VK_RIGHT))
        {
            m_cameraYaw += m_config.keyLookSpeed * deltaTime;
        }
        if (IsKeyDown(VK_UP))
        {
            m_cameraPitch += m_config.keyLookSpeed * deltaTime;
        }
        if (IsKeyDown(VK_DOWN))
        {
            m_cameraPitch -= m_config.keyLookSpeed * deltaTime;
        }
        m_cameraPitch = std::clamp(m_cameraPitch, -maxPitch, maxPitch);

        Vec3 movement{};
        const Vec3 planarForward = GetCameraPlanarForward();
        const Vec3 planarRight = Normalize(Cross({0.0f, 1.0f, 0.0f}, planarForward));
        if (IsKeyDown('W'))
        {
            movement += planarForward;
        }
        if (IsKeyDown('S'))
        {
            movement += planarForward * -1.0f;
        }
        if (IsKeyDown('D'))
        {
            movement += planarRight;
        }
        if (IsKeyDown('A'))
        {
            movement += planarRight * -1.0f;
        }
        if (IsKeyDown('E') || IsKeyDown(VK_SPACE))
        {
            movement += {0.0f, 1.0f, 0.0f};
        }
        if (IsKeyDown('Q') || IsKeyDown(VK_CONTROL))
        {
            movement += {0.0f, -1.0f, 0.0f};
        }

        if (Length(movement) > 0.0f)
        {
            const float moveSpeed = IsKeyDown(VK_SHIFT) ? m_config.fastMoveSpeed : m_config.moveSpeed;
            m_cameraPosition += Normalize(movement) * (moveSpeed * deltaTime);
        }
    }

    void CreateWindowAndShow()
    {
        const HINSTANCE instance = GetModuleHandleW(nullptr);
        const wchar_t* className = L"VulkanPathTracerWindowClass";

        WNDCLASSEXW windowClass{};
        windowClass.cbSize = sizeof(windowClass);
        windowClass.hInstance = instance;
        windowClass.lpfnWndProc = WindowProc;
        windowClass.lpszClassName = className;
        windowClass.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));

        const ATOM atom = RegisterClassExW(&windowClass);
        if (atom == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
        {
            throw std::runtime_error("Failed to register window class.");
        }

        RECT rect{0, 0, static_cast<LONG>(m_config.width), static_cast<LONG>(m_config.height)};
        const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        ThrowIfFalse(AdjustWindowRect(&rect, style, FALSE), "Failed to size window.");

        m_window = CreateWindowExW(0,
                                   className,
                                   L"Vulkan Path Tracer - FPS: measuring...",
                                   style,
                                   CW_USEDEFAULT,
                                   CW_USEDEFAULT,
                                   rect.right - rect.left,
                                   rect.bottom - rect.top,
                                   nullptr,
                                   nullptr,
                                   instance,
                                   nullptr);
        if (m_window == nullptr)
        {
            throw std::runtime_error("Failed to create window.");
        }

        ShowWindow(m_window, SW_SHOWDEFAULT);
        UpdateWindow(m_window);
        std::printf("[Config] Edit %s and save to hot-reload tuning.\n", m_configPath.string().c_str());
        std::puts("[Config] width, height, and frameCount are loaded from JSON at startup.");
        std::puts("[Controls] Hold RMB to look. Move with WASD, rise/fall with Q/E or Ctrl/Space, Shift to boost, R to reset.");
    }

    void CreateInstance()
    {
        const std::array<const char*, 2> extensions = {
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        };

        VkApplicationInfo applicationInfo{};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "Vulkan Path Tracer";
        applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        applicationInfo.pEngineName = "None";
        applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        applicationInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &applicationInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        ThrowVk(vkCreateInstance(&createInfo, nullptr, &m_instance), "Failed to create Vulkan instance");
    }

    void CreateSurface()
    {
        VkWin32SurfaceCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        createInfo.hinstance = GetModuleHandleW(nullptr);
        createInfo.hwnd = m_window;
        ThrowVk(vkCreateWin32SurfaceKHR(m_instance, &createInfo, nullptr, &m_surface),
                "Failed to create Win32 surface");
    }

    QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) const
    {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, families.data());

        for (uint32_t i = 0; i < queueFamilyCount; ++i)
        {
            if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
            {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupported);
            if (presentSupported == VK_TRUE)
            {
                indices.presentFamily = i;
            }

            if (indices.IsComplete())
            {
                break;
            }
        }

        return indices;
    }

    SwapchainSupport QuerySwapchainSupport(VkPhysicalDevice device) const
    {
        SwapchainSupport support{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_surface, &support.capabilities);

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, nullptr);
        support.formats.resize(formatCount);
        if (formatCount > 0)
        {
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, support.formats.data());
        }

        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &presentModeCount, nullptr);
        support.presentModes.resize(presentModeCount);
        if (presentModeCount > 0)
        {
            vkGetPhysicalDeviceSurfacePresentModesKHR(device,
                                                      m_surface,
                                                      &presentModeCount,
                                                      support.presentModes.data());
        }

        return support;
    }

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR GetRayTracingPipelineProperties(VkPhysicalDevice device) const
    {
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;

        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &properties;
        vkGetPhysicalDeviceProperties2(device, &properties2);
        return properties;
    }

    bool IsDeviceSuitable(VkPhysicalDevice device)
    {
        const auto queueFamilies = FindQueueFamilies(device);
        if (!queueFamilies.IsComplete())
        {
            return false;
        }

        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());

        bool hasSwapchain = false;
        bool hasAccelerationStructure = false;
        bool hasRayTracingPipeline = false;
        bool hasDeferredHostOperations = false;
        bool hasBufferDeviceAddress = false;
        bool hasShaderNonSemanticInfo = false;
        for (const auto& extension : extensions)
        {
            if (std::strcmp(extension.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
            {
                hasSwapchain = true;
            }
            if (std::strcmp(extension.extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0)
            {
                hasAccelerationStructure = true;
            }
            if (std::strcmp(extension.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0)
            {
                hasRayTracingPipeline = true;
            }
            if (std::strcmp(extension.extensionName, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) == 0)
            {
                hasDeferredHostOperations = true;
            }
            if (std::strcmp(extension.extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0)
            {
                hasBufferDeviceAddress = true;
            }
            if (std::strcmp(extension.extensionName, VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME) == 0)
            {
                hasShaderNonSemanticInfo = true;
            }
            if (hasSwapchain && hasAccelerationStructure && hasRayTracingPipeline && hasDeferredHostOperations
                && hasBufferDeviceAddress && hasShaderNonSemanticInfo)
            {
                break;
            }
        }
        const bool hasRequiredExtensions = hasSwapchain && hasAccelerationStructure && hasRayTracingPipeline
                                           && hasDeferredHostOperations && hasBufferDeviceAddress && hasShaderNonSemanticInfo;
        if (!hasRequiredExtensions)
        {
            return false;
        }

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{};
        bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};
        accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{};
        rayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        features2.pNext = &bufferDeviceAddressFeatures;
        bufferDeviceAddressFeatures.pNext = &accelerationStructureFeatures;
        accelerationStructureFeatures.pNext = &rayTracingPipelineFeatures;
        vkGetPhysicalDeviceFeatures2(device, &features2);

        if (bufferDeviceAddressFeatures.bufferDeviceAddress != VK_TRUE
            || accelerationStructureFeatures.accelerationStructure != VK_TRUE
            || rayTracingPipelineFeatures.rayTracingPipeline != VK_TRUE)
        {
            return false;
        }

        const auto rayTracingProperties = GetRayTracingPipelineProperties(device);
        if (rayTracingProperties.maxRayRecursionDepth < m_config.maxBounces
            || rayTracingProperties.shaderGroupHandleSize == 0)
        {
            return false;
        }

        const auto swapchainSupport = QuerySwapchainSupport(device);
        if (swapchainSupport.formats.empty() || swapchainSupport.presentModes.empty())
        {
            return false;
        }

        return (swapchainSupport.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
    }

    void PickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        ThrowVk(vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr),
                "Failed to enumerate Vulkan physical devices");
        if (deviceCount == 0)
        {
            throw std::runtime_error("No Vulkan-capable GPU found.");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        ThrowVk(vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data()),
                "Failed to enumerate Vulkan physical devices");

        for (VkPhysicalDevice device : devices)
        {
            if (IsDeviceSuitable(device))
            {
                m_physicalDevice = device;
                m_queueFamilies = FindQueueFamilies(device);
                m_rayTracingPipelineProperties = GetRayTracingPipelineProperties(device);
                break;
            }
        }

        if (m_physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("No Vulkan device supports storage-image swapchains for this app.");
        }
    }

    void CreateLogicalDevice()
    {
        const float queuePriority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        const std::array<uint32_t, 2> uniqueFamilies = {
            m_queueFamilies.graphicsFamily.value(),
            m_queueFamilies.presentFamily.value(),
        };

        std::vector<uint32_t> familyList;
        for (uint32_t family : uniqueFamilies)
        {
            if (std::find(familyList.begin(), familyList.end(), family) == familyList.end())
            {
                familyList.push_back(family);
            }
        }

        for (uint32_t family : familyList)
        {
            VkDeviceQueueCreateInfo queueInfo{};
            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueFamilyIndex = family;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &queuePriority;
            queueInfos.push_back(queueInfo);
        }

        const std::array<const char*, 6> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
        };

        VkPhysicalDeviceFeatures2 deviceFeatures{};
        deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{};
        bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};
        accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        accelerationStructureFeatures.accelerationStructure = VK_TRUE;
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{};
        rayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
        deviceFeatures.pNext = &bufferDeviceAddressFeatures;
        bufferDeviceAddressFeatures.pNext = &accelerationStructureFeatures;
        accelerationStructureFeatures.pNext = &rayTracingPipelineFeatures;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = &deviceFeatures;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
        createInfo.pQueueCreateInfos = queueInfos.data();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        ThrowVk(vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device),
                "Failed to create Vulkan logical device");

        vkGetDeviceQueue(m_device, m_queueFamilies.graphicsFamily.value(), 0, &m_graphicsQueue);
        vkGetDeviceQueue(m_device, m_queueFamilies.presentFamily.value(), 0, &m_presentQueue);

        m_vkGetBufferDeviceAddressKHR = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(
            vkGetDeviceProcAddr(m_device, "vkGetBufferDeviceAddressKHR"));
        m_vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
            vkGetDeviceProcAddr(m_device, "vkCreateAccelerationStructureKHR"));
        m_vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
            vkGetDeviceProcAddr(m_device, "vkDestroyAccelerationStructureKHR"));
        m_vkGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
            vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureBuildSizesKHR"));
        m_vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
            vkGetDeviceProcAddr(m_device, "vkCmdBuildAccelerationStructuresKHR"));
        m_vkGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
            vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureDeviceAddressKHR"));
        m_vkCreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
            vkGetDeviceProcAddr(m_device, "vkCreateRayTracingPipelinesKHR"));
        m_vkGetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
            vkGetDeviceProcAddr(m_device, "vkGetRayTracingShaderGroupHandlesKHR"));
        m_vkCmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(
            vkGetDeviceProcAddr(m_device, "vkCmdTraceRaysKHR"));

        if (m_vkGetBufferDeviceAddressKHR == nullptr || m_vkCreateAccelerationStructureKHR == nullptr
            || m_vkDestroyAccelerationStructureKHR == nullptr
            || m_vkGetAccelerationStructureBuildSizesKHR == nullptr
            || m_vkCmdBuildAccelerationStructuresKHR == nullptr
            || m_vkGetAccelerationStructureDeviceAddressKHR == nullptr
            || m_vkCreateRayTracingPipelinesKHR == nullptr
            || m_vkGetRayTracingShaderGroupHandlesKHR == nullptr
            || m_vkCmdTraceRaysKHR == nullptr)
        {
            throw std::runtime_error("Missing required Vulkan ray tracing extension entry points.");
        }
    }

    uint32_t FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties) const
    {
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
        {
            if ((typeBits & (1u << i)) != 0
                && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable Vulkan memory type.");
    }

    BufferAllocation CreateBuffer(VkDeviceSize size,
                                  VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags properties,
                                  bool enableDeviceAddress) const
    {
        BufferAllocation allocation{};
        allocation.size = size;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ThrowVk(vkCreateBuffer(m_device, &bufferInfo, nullptr, &allocation.buffer), "Failed to create buffer");

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(m_device, allocation.buffer, &memoryRequirements);

        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        if (enableDeviceAddress)
        {
            flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties);
        if (enableDeviceAddress)
        {
            allocateInfo.pNext = &flagsInfo;
        }

        ThrowVk(vkAllocateMemory(m_device, &allocateInfo, nullptr, &allocation.memory), "Failed to allocate buffer memory");
        ThrowVk(vkBindBufferMemory(m_device, allocation.buffer, allocation.memory, 0), "Failed to bind buffer memory");
        return allocation;
    }

    void DestroyBuffer(BufferAllocation& allocation)
    {
        if (allocation.buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(m_device, allocation.buffer, nullptr);
            allocation.buffer = VK_NULL_HANDLE;
        }
        if (allocation.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(m_device, allocation.memory, nullptr);
            allocation.memory = VK_NULL_HANDLE;
        }
        allocation.size = 0;
    }

    VkDeviceAddress GetBufferDeviceAddress(VkBuffer buffer) const
    {
        VkBufferDeviceAddressInfo addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addressInfo.buffer = buffer;
        return m_vkGetBufferDeviceAddressKHR(m_device, &addressInfo);
    }

    VkDeviceAddress GetAccelerationStructureDeviceAddress(VkAccelerationStructureKHR structure) const
    {
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addressInfo.accelerationStructure = structure;
        return m_vkGetAccelerationStructureDeviceAddressKHR(m_device, &addressInfo);
    }

    VkCommandBuffer BeginSingleTimeCommands() const
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        ThrowVk(vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer),
                "Failed to allocate one-time command buffer");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        ThrowVk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "Failed to begin one-time command buffer");
        return commandBuffer;
    }

    void EndSingleTimeCommands(VkCommandBuffer commandBuffer) const
    {
        ThrowVk(vkEndCommandBuffer(commandBuffer), "Failed to end one-time command buffer");

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        ThrowVk(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE),
                "Failed to submit one-time command buffer");
        ThrowVk(vkQueueWaitIdle(m_graphicsQueue), "Failed to wait for one-time command submission");
        vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
    }

    void UploadToBuffer(const BufferAllocation& allocation, const void* data, size_t dataSize) const
    {
        if (dataSize > static_cast<size_t>(allocation.size))
        {
            throw std::runtime_error("Upload exceeds destination buffer size.");
        }
        void* mapped = nullptr;
        ThrowVk(vkMapMemory(m_device, allocation.memory, 0, allocation.size, 0, &mapped), "Failed to map buffer memory");
        std::memcpy(mapped, data, dataSize);
        vkUnmapMemory(m_device, allocation.memory);
    }

    AccelerationStructureAllocation CreateAccelerationStructure(VkAccelerationStructureTypeKHR type,
                                                                VkDeviceSize size) const
    {
        AccelerationStructureAllocation allocation{};
        allocation.buffer = CreateBuffer(size,
                                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                             | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                         true);

        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.type = type;
        createInfo.size = size;
        createInfo.buffer = allocation.buffer.buffer;
        ThrowVk(m_vkCreateAccelerationStructureKHR(m_device, &createInfo, nullptr, &allocation.handle),
                "Failed to create acceleration structure");
        return allocation;
    }

    void CreateRayTracingScene()
    {
        if (m_commandPool == VK_NULL_HANDLE)
        {
            throw std::runtime_error("Command pool must be created before ray tracing scene setup.");
        }

        CreateGeometryResourcesFromConfig();
        CreateSceneBuffers();
        UploadSceneDataFromConfig();
        RebuildTopLevelAccelerationStructure();
    }

    VkSurfaceFormatKHR ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const
    {
        for (const auto& format : formats)
        {
            if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return format;
            }
        }

        return formats.front();
    }

    VkPresentModeKHR ChoosePresentMode(const std::vector<VkPresentModeKHR>& modes) const
    {
        for (VkPresentModeKHR mode : modes)
        {
            if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
            {
                return mode;
            }
        }

        for (VkPresentModeKHR mode : modes)
        {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return mode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    void CreateSwapchain()
    {
        const auto support = QuerySwapchainSupport(m_physicalDevice);
        const auto surfaceFormat = ChooseSurfaceFormat(support.formats);
        const auto presentMode = ChoosePresentMode(support.presentModes);

        m_swapchainExtent = {m_config.width, m_config.height};
        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 && m_config.frameCount > support.capabilities.maxImageCount)
        {
            throw std::runtime_error("Configured frameCount exceeds the swapchain image limit for this surface.");
        }
        if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
        {
            imageCount = support.capabilities.maxImageCount;
        }
        imageCount = std::max(imageCount, m_config.frameCount);

        const uint32_t queueFamilyIndices[] = {
            m_queueFamilies.graphicsFamily.value(),
            m_queueFamilies.presentFamily.value(),
        };

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = m_surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = m_swapchainExtent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT;
        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        if (m_queueFamilies.graphicsFamily != m_queueFamilies.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        ThrowVk(vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain),
                "Failed to create Vulkan swapchain");

        m_swapchainFormat = surfaceFormat.format;

        uint32_t actualImageCount = 0;
        ThrowVk(vkGetSwapchainImagesKHR(m_device, m_swapchain, &actualImageCount, nullptr),
                "Failed to query swapchain image count");
        m_swapchainImages.resize(actualImageCount);
        ThrowVk(vkGetSwapchainImagesKHR(m_device, m_swapchain, &actualImageCount, m_swapchainImages.data()),
                "Failed to get swapchain images");
        m_swapchainImageViews.resize(actualImageCount);
        m_swapchainLayouts.assign(actualImageCount, VK_IMAGE_LAYOUT_UNDEFINED);

        for (size_t i = 0; i < m_swapchainImages.size(); ++i)
        {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = m_swapchainImages[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = m_swapchainFormat;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            ThrowVk(vkCreateImageView(m_device, &viewInfo, nullptr, &m_swapchainImageViews[i]),
                    "Failed to create swapchain image view");
        }
    }

    void CreateDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

        VkDescriptorSetLayoutBinding accelerationBinding{};
        accelerationBinding.binding = 1;
        accelerationBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        accelerationBinding.descriptorCount = 1;
        accelerationBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        VkDescriptorSetLayoutBinding sceneBinding{};
        sceneBinding.binding = 2;
        sceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sceneBinding.descriptorCount = 1;
        sceneBinding.stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        VkDescriptorSetLayoutBinding instanceBinding{};
        instanceBinding.binding = 3;
        instanceBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        instanceBinding.descriptorCount = 1;
        instanceBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        VkDescriptorSetLayoutBinding materialBinding{};
        materialBinding.binding = 4;
        materialBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        materialBinding.descriptorCount = 1;
        materialBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        VkDescriptorSetLayoutBinding vertexBinding{};
        vertexBinding.binding = 5;
        vertexBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        vertexBinding.descriptorCount = 1;
        vertexBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        VkDescriptorSetLayoutBinding indexBinding{};
        indexBinding.binding = 6;
        indexBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        indexBinding.descriptorCount = 1;
        indexBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        const std::array<VkDescriptorSetLayoutBinding, 7> bindings = {
            binding,
            accelerationBinding,
            sceneBinding,
            instanceBinding,
            materialBinding,
            vertexBinding,
            indexBinding,
        };

        VkDescriptorSetLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        createInfo.pBindings = bindings.data();
        ThrowVk(vkCreateDescriptorSetLayout(m_device, &createInfo, nullptr, &m_descriptorSetLayout),
                "Failed to create descriptor set layout");
    }

    VkShaderModule CreateShaderModule(const std::vector<char>& bytecode)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = bytecode.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(bytecode.data());

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        ThrowVk(vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule),
                "Failed to create shader module");
        return shaderModule;
    }

    BufferAllocation CreateShaderBindingTableBuffer(const void* data, VkDeviceSize size) const
    {
        BufferAllocation allocation = CreateBuffer(size,
                                                   VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR
                                                       | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                   true);
        UploadToBuffer(allocation, data, static_cast<size_t>(size));
        return allocation;
    }

    void CreatePipeline()
    {
        if (m_rayTracingPipelineProperties.maxRayRecursionDepth < m_config.maxBounces)
        {
            throw std::runtime_error("The selected GPU does not support the required ray recursion depth.");
        }

        const auto raygenBytecode = LoadBinaryFile(L"path_tracer.rgen.spv");
        const auto missBytecode = LoadBinaryFile(L"path_tracer.rmiss.spv");
        const auto closestHitBytecode = LoadBinaryFile(L"path_tracer.rchit.spv");
        VkShaderModule raygenModule = CreateShaderModule(raygenBytecode);
        VkShaderModule missModule = CreateShaderModule(missBytecode);
        VkShaderModule closestHitModule = CreateShaderModule(closestHitBytecode);

        const std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages = {{
            {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                nullptr,
                0,
                VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                raygenModule,
                "main",
                nullptr,
            },
            {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                nullptr,
                0,
                VK_SHADER_STAGE_MISS_BIT_KHR,
                missModule,
                "main",
                nullptr,
            },
            {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                nullptr,
                0,
                VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                closestHitModule,
                "main",
                nullptr,
            },
        }};

        const std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> shaderGroups = {{
            {
                VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                nullptr,
                VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                0,
                VK_SHADER_UNUSED_KHR,
                VK_SHADER_UNUSED_KHR,
                VK_SHADER_UNUSED_KHR,
                nullptr,
            },
            {
                VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                nullptr,
                VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                1,
                VK_SHADER_UNUSED_KHR,
                VK_SHADER_UNUSED_KHR,
                VK_SHADER_UNUSED_KHR,
                nullptr,
            },
            {
                VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                nullptr,
                VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                VK_SHADER_UNUSED_KHR,
                2,
                VK_SHADER_UNUSED_KHR,
                VK_SHADER_UNUSED_KHR,
                nullptr,
            },
        }};

        VkPushConstantRange pushRange{};
        pushRange.stageFlags =
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_descriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
        ThrowVk(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipelineLayout),
                "Failed to create pipeline layout");

        VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.groupCount = static_cast<uint32_t>(shaderGroups.size());
        pipelineInfo.pGroups = shaderGroups.data();
        pipelineInfo.maxPipelineRayRecursionDepth = m_rayTracingPipelineProperties.maxRayRecursionDepth;
        pipelineInfo.layout = m_pipelineLayout;
        ThrowVk(m_vkCreateRayTracingPipelinesKHR(m_device,
                                                 VK_NULL_HANDLE,
                                                 VK_NULL_HANDLE,
                                                 1,
                                                 &pipelineInfo,
                                                 nullptr,
                                                 &m_rayTracingPipeline),
                "Failed to create ray tracing pipeline");

        vkDestroyShaderModule(m_device, closestHitModule, nullptr);
        vkDestroyShaderModule(m_device, missModule, nullptr);
        vkDestroyShaderModule(m_device, raygenModule, nullptr);
    }

    void CreateShaderBindingTables()
    {
        constexpr uint32_t kShaderGroupCount = 3;

        const uint32_t handleSize = m_rayTracingPipelineProperties.shaderGroupHandleSize;
        const VkDeviceSize handleSizeAligned =
            AlignUp(handleSize, m_rayTracingPipelineProperties.shaderGroupHandleAlignment);
        const VkDeviceSize recordSize =
            AlignUp(handleSizeAligned, m_rayTracingPipelineProperties.shaderGroupBaseAlignment);

        std::vector<uint8_t> shaderGroupHandles(static_cast<size_t>(handleSize) * kShaderGroupCount);
        ThrowVk(m_vkGetRayTracingShaderGroupHandlesKHR(m_device,
                                                       m_rayTracingPipeline,
                                                       0,
                                                       kShaderGroupCount,
                                                       shaderGroupHandles.size(),
                                                       shaderGroupHandles.data()),
                "Failed to fetch ray tracing shader group handles");

        auto createRegion = [&](uint32_t groupIndex,
                                BufferAllocation& buffer,
                                VkStridedDeviceAddressRegionKHR& region)
        {
            std::vector<uint8_t> record(static_cast<size_t>(recordSize), 0);
            std::memcpy(record.data(),
                        shaderGroupHandles.data() + static_cast<size_t>(groupIndex) * handleSize,
                        handleSize);

            buffer = CreateShaderBindingTableBuffer(record.data(), recordSize);
            region.deviceAddress = GetBufferDeviceAddress(buffer.buffer);
            region.stride = recordSize;
            region.size = recordSize;
        };

        createRegion(0, m_raygenShaderBindingTable, m_raygenShaderBindingTableRegion);
        createRegion(1, m_missShaderBindingTable, m_missShaderBindingTableRegion);
        createRegion(2, m_hitShaderBindingTable, m_hitShaderBindingTableRegion);
        m_callableShaderBindingTableRegion = {};
    }

    void CreateDescriptorSets()
    {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSize.descriptorCount = static_cast<uint32_t>(m_swapchainImageViews.size());
        VkDescriptorPoolSize accelerationPoolSize{};
        accelerationPoolSize.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        accelerationPoolSize.descriptorCount = static_cast<uint32_t>(m_swapchainImageViews.size());
        VkDescriptorPoolSize scenePoolSize{};
        scenePoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        scenePoolSize.descriptorCount = static_cast<uint32_t>(m_swapchainImageViews.size());
        VkDescriptorPoolSize storageBufferPoolSize{};
        storageBufferPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        storageBufferPoolSize.descriptorCount = static_cast<uint32_t>(m_swapchainImageViews.size() * 4);
        const std::array<VkDescriptorPoolSize, 4> poolSizes = {
            poolSize,
            accelerationPoolSize,
            scenePoolSize,
            storageBufferPoolSize,
        };

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = static_cast<uint32_t>(m_swapchainImageViews.size());
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        ThrowVk(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool),
                "Failed to create descriptor pool");

        std::vector<VkDescriptorSetLayout> layouts(m_swapchainImageViews.size(), m_descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        allocInfo.pSetLayouts = layouts.data();

        m_descriptorSets.resize(layouts.size());
        ThrowVk(vkAllocateDescriptorSets(m_device, &allocInfo, m_descriptorSets.data()),
                "Failed to allocate descriptor sets");
        UpdateDescriptorSetContents();
    }

    void CreateCommandPool()
    {
        VkCommandPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        createInfo.queueFamilyIndex = m_queueFamilies.graphicsFamily.value();
        ThrowVk(vkCreateCommandPool(m_device, &createInfo, nullptr, &m_commandPool),
                "Failed to create command pool");
    }

    void CreateCommandBuffers()
    {
        m_commandBuffers.resize(m_config.frameCount);
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());
        ThrowVk(vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()),
                "Failed to allocate command buffers");
    }

    void CreateSyncObjects()
    {
        m_frames.resize(m_config.frameCount);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (auto& frame : m_frames)
        {
            ThrowVk(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &frame.imageAvailable),
                    "Failed to create image-available semaphore");
            ThrowVk(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &frame.renderFinished),
                    "Failed to create render-finished semaphore");
            ThrowVk(vkCreateFence(m_device, &fenceInfo, nullptr, &frame.inFlight),
                    "Failed to create frame fence");
        }
    }

    PushConstants BuildPushConstants() const
    {
        PushConstants constants{};
        constants.cameraPositionFrame[0] = m_cameraPosition.x;
        constants.cameraPositionFrame[1] = m_cameraPosition.y;
        constants.cameraPositionFrame[2] = m_cameraPosition.z;
        constants.cameraPositionFrame[3] = static_cast<float>(m_frameIndex);

        const Vec3 forward = GetCameraForward();
        const Vec3 right = Normalize(Cross({0.0f, 1.0f, 0.0f}, forward));
        const Vec3 up = Normalize(Cross(forward, right));

        constants.cameraForwardSamples[0] = forward.x;
        constants.cameraForwardSamples[1] = forward.y;
        constants.cameraForwardSamples[2] = forward.z;
        constants.cameraForwardSamples[3] = static_cast<float>(m_config.samplesPerPixel);
        constants.cameraRightBounces[0] = right.x;
        constants.cameraRightBounces[1] = right.y;
        constants.cameraRightBounces[2] = right.z;
        constants.cameraRightBounces[3] = static_cast<float>(m_config.maxBounces);
        constants.cameraUpTanHalfFovY[0] = up.x;
        constants.cameraUpTanHalfFovY[1] = up.y;
        constants.cameraUpTanHalfFovY[2] = up.z;
        constants.cameraUpTanHalfFovY[3] = std::tan(m_config.fovYDegrees * 0.5f * kPi / 180.0f);
        constants.skyBottomExposure[0] = m_config.skyBottomColor[0];
        constants.skyBottomExposure[1] = m_config.skyBottomColor[1];
        constants.skyBottomExposure[2] = m_config.skyBottomColor[2];
        constants.skyBottomExposure[3] = m_config.skyExposure;
        constants.skyTopAspect[0] = m_config.skyTopColor[0];
        constants.skyTopAspect[1] = m_config.skyTopColor[1];
        constants.skyTopAspect[2] = m_config.skyTopColor[2];
        constants.skyTopAspect[3] =
            static_cast<float>(m_swapchainExtent.width) / static_cast<float>(m_swapchainExtent.height);

        constants.imageSize[0] = m_swapchainExtent.width;
        constants.imageSize[1] = m_swapchainExtent.height;
        return constants;
    }

    void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        ThrowVk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "Failed to begin command buffer");

        VkImageMemoryBarrier toGeneral{};
        toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toGeneral.oldLayout = m_swapchainLayouts[imageIndex];
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = m_swapchainImages[imageIndex];
        toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toGeneral.subresourceRange.levelCount = 1;
        toGeneral.subresourceRange.layerCount = 1;
        toGeneral.srcAccessMask = 0;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &toGeneral);

        const PushConstants pushConstants = BuildPushConstants();
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rayTracingPipeline);
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                m_pipelineLayout,
                                0,
                                1,
                                &m_descriptorSets[imageIndex],
                                0,
                                nullptr);
        vkCmdPushConstants(commandBuffer,
                           m_pipelineLayout,
                           VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR
                               | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                           0,
                           sizeof(PushConstants),
                           &pushConstants);
        m_vkCmdTraceRaysKHR(commandBuffer,
                            &m_raygenShaderBindingTableRegion,
                            &m_missShaderBindingTableRegion,
                            &m_hitShaderBindingTableRegion,
                            &m_callableShaderBindingTableRegion,
                            m_swapchainExtent.width,
                            m_swapchainExtent.height,
                            1);

        VkImageMemoryBarrier toPresent = toGeneral;
        toPresent.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toPresent.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        toPresent.dstAccessMask = 0;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &toPresent);

        ThrowVk(vkEndCommandBuffer(commandBuffer), "Failed to end command buffer");
        m_swapchainLayouts[imageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }

    void RenderFrame()
    {
        FrameResources& frame = m_frames[m_currentFrame];
        ThrowVk(vkWaitForFences(m_device, 1, &frame.inFlight, VK_TRUE, UINT64_MAX), "Failed to wait for fence");

        uint32_t imageIndex = 0;
        const VkResult acquire = vkAcquireNextImageKHR(m_device,
                                                       m_swapchain,
                                                       UINT64_MAX,
                                                       frame.imageAvailable,
                                                       VK_NULL_HANDLE,
                                                       &imageIndex);
        if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR)
        {
            ThrowVk(acquire, "Failed to acquire swapchain image");
        }

        ThrowVk(vkResetFences(m_device, 1, &frame.inFlight), "Failed to reset fence");
        ThrowVk(vkResetCommandBuffer(m_commandBuffers[m_currentFrame], 0), "Failed to reset command buffer");
        RecordCommandBuffer(m_commandBuffers[m_currentFrame], imageIndex);

        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &frame.imageAvailable;
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffers[m_currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &frame.renderFinished;
        ThrowVk(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, frame.inFlight), "Failed to submit command buffer");

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &frame.renderFinished;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &m_swapchain;
        presentInfo.pImageIndices = &imageIndex;
        const VkResult present = vkQueuePresentKHR(m_presentQueue, &presentInfo);
        if (present != VK_SUCCESS && present != VK_SUBOPTIMAL_KHR)
        {
            ThrowVk(present, "Failed to present swapchain image");
        }

        ++m_frameIndex;
        m_currentFrame = (m_currentFrame + 1) % static_cast<uint32_t>(m_frames.size());
    }

    void UpdateWindowTitle(double fps, double frameMs)
    {
        wchar_t buffer[256]{};
        std::swprintf(buffer,
                      sizeof(buffer) / sizeof(buffer[0]),
                      L"Vulkan Path Tracer - %.1f FPS (%.2f ms)",
                      fps,
                      frameMs);
        SetWindowTextW(m_window, buffer);
    }

    void MessageLoop()
    {
        using Clock = std::chrono::steady_clock;
        MSG message{};
        auto statsStart = Clock::now();
        auto previousFrameStart = statsStart;
        uint32_t framesSinceUpdate = 0;

        while (true)
        {
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
            {
                if (message.message == WM_QUIT)
                {
                    return;
                }

                TranslateMessage(&message);
                DispatchMessageW(&message);
            }

            const auto frameStart = Clock::now();
            ReloadRuntimeConfigIfNeeded();
            const double deltaSeconds = std::chrono::duration<double>(frameStart - previousFrameStart).count();
            previousFrameStart = frameStart;
            UpdateCamera(deltaSeconds);
            RenderFrame();
            const auto frameEnd = Clock::now();

            ++framesSinceUpdate;
            const double elapsedSeconds = std::chrono::duration<double>(frameEnd - statsStart).count();
            if (elapsedSeconds >= 1.0)
            {
                const double fps = static_cast<double>(framesSinceUpdate) / elapsedSeconds;
                const double frameMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
                UpdateWindowTitle(fps, frameMs);
                std::printf("[Vulkan] %.1f FPS (%.2f ms)\n", fps, frameMs);
                framesSinceUpdate = 0;
                statsStart = frameEnd;
            }
        }
    }

    void Cleanup()
    {
        if (m_device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(m_device);
        }

        for (auto& frame : m_frames)
        {
            if (frame.inFlight != VK_NULL_HANDLE) vkDestroyFence(m_device, frame.inFlight, nullptr);
            if (frame.renderFinished != VK_NULL_HANDLE) vkDestroySemaphore(m_device, frame.renderFinished, nullptr);
            if (frame.imageAvailable != VK_NULL_HANDLE) vkDestroySemaphore(m_device, frame.imageAvailable, nullptr);
        }

        if (!m_commandBuffers.empty() && m_commandPool != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(m_device,
                                 m_commandPool,
                                 static_cast<uint32_t>(m_commandBuffers.size()),
                                 m_commandBuffers.data());
        }
        if (m_commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        if (m_descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        if (m_rayTracingPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_device, m_rayTracingPipeline, nullptr);
        if (m_pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        if (m_descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
        DestroyGeometryResources();
        DestroySceneBuffers();
        DestroyBuffer(m_raygenShaderBindingTable);
        DestroyBuffer(m_missShaderBindingTable);
        DestroyBuffer(m_hitShaderBindingTable);
        for (VkImageView view : m_swapchainImageViews)
        {
            if (view != VK_NULL_HANDLE) vkDestroyImageView(m_device, view, nullptr);
        }
        if (m_swapchain != VK_NULL_HANDLE) vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        if (m_device != VK_NULL_HANDLE) vkDestroyDevice(m_device, nullptr);
        if (m_surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        if (m_instance != VK_NULL_HANDLE) vkDestroyInstance(m_instance, nullptr);
        if (m_window != nullptr) DestroyWindow(m_window);
    }

    struct FrameResources
    {
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
    };

    HWND m_window = nullptr;

    VkInstance m_instance = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue m_presentQueue = VK_NULL_HANDLE;
    QueueFamilyIndices m_queueFamilies;

    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D m_swapchainExtent{};
    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<VkImageLayout> m_swapchainLayouts;

    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_rayTracingPipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_descriptorSets;

    BufferAllocation m_vertexBuffer{};
    BufferAllocation m_indexBuffer{};
    BufferAllocation m_instanceBuffer{};
    BufferAllocation m_sceneDataBuffer{};
    BufferAllocation m_instanceDataBuffer{};
    BufferAllocation m_materialDataBuffer{};
    BufferAllocation m_raygenShaderBindingTable{};
    BufferAllocation m_missShaderBindingTable{};
    BufferAllocation m_hitShaderBindingTable{};
    std::vector<ModelAccelerationStructure> m_modelAccelerationStructures;
    AccelerationStructureAllocation m_tlas{};
    VkStridedDeviceAddressRegionKHR m_raygenShaderBindingTableRegion{};
    VkStridedDeviceAddressRegionKHR m_missShaderBindingTableRegion{};
    VkStridedDeviceAddressRegionKHR m_hitShaderBindingTableRegion{};
    VkStridedDeviceAddressRegionKHR m_callableShaderBindingTableRegion{};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rayTracingPipelineProperties{};

    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_commandBuffers;
    std::vector<FrameResources> m_frames;
    uint32_t m_currentFrame = 0;
    uint64_t m_frameIndex = 0;
    RuntimeConfig m_config{};
    std::filesystem::path m_configPath;
    std::filesystem::file_time_type m_configLastWriteTime{};
    std::chrono::steady_clock::time_point m_lastConfigPollTime{};
    Vec3 m_cameraPosition{};
    float m_cameraYaw = 0.0f;
    float m_cameraPitch = 0.0f;
    bool m_resetCameraKeyDown = false;
    bool m_mouseLookActive = false;
    POINT m_lastMousePosition{};

    PFN_vkGetBufferDeviceAddressKHR m_vkGetBufferDeviceAddressKHR = nullptr;
    PFN_vkCreateAccelerationStructureKHR m_vkCreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR m_vkDestroyAccelerationStructureKHR = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR m_vkGetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR m_vkCmdBuildAccelerationStructuresKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR m_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR m_vkCreateRayTracingPipelinesKHR = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR m_vkGetRayTracingShaderGroupHandlesKHR = nullptr;
    PFN_vkCmdTraceRaysKHR m_vkCmdTraceRaysKHR = nullptr;
};

void RunVulkanPathTracer()
{
    VulkanPathTracer app;
    app.Run();
}
