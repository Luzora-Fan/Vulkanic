#include "VulkanPathTracer.h"

#include <cstdio>
#include <exception>

int main()
{
    try
    {
        RunVulkanPathTracer();
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "[ERROR] %s\n", error.what());
        return 1;
    }
}

// To build this project run this command:
// powershell -ExecutionPolicy Bypass -File .\build.ps1 -Clean