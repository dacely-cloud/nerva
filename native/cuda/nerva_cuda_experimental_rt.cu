#include "nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <math.h>
#include <new>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <vector>

#if __has_include(<optix_host.h>) && __has_include(<optix_stubs.h>) && \
    __has_include(<optix_stack_size.h>) && __has_include(<nvrtc.h>)
#define NERVA_HAVE_OPTIX_HEADERS 1
#include <cuda.h>
#include <nvrtc.h>
#include <optix_function_table_definition.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#else
#define NERVA_HAVE_OPTIX_HEADERS 0
#endif

#if __has_include(<vulkan/vulkan.h>)
#include <vulkan/vulkan.h>
#define NERVA_HAVE_VULKAN_HEADERS 1
#else
#define NERVA_HAVE_VULKAN_HEADERS 0
#endif

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxInitBlocks = 4096;

uint64_t checked_mul_u64(uint64_t lhs, uint64_t rhs) {
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  if (lhs > UINT64_MAX / rhs) {
    return UINT64_MAX;
  }
  return lhs * rhs;
}

uint64_t elapsed_ns(cudaEvent_t start, cudaEvent_t stop) {
  float elapsed_ms = 0.0f;
  cudaError_t err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess || elapsed_ms <= 0.0f) {
    return 0;
  }
  const uint64_t ns = static_cast<uint64_t>(elapsed_ms * 1000000.0f);
  return ns == 0 ? 1 : ns;
}

uint64_t speedup_x1000(uint64_t baseline_ns, uint64_t candidate_ns) {
  if (candidate_ns == 0 || baseline_ns > UINT64_MAX / 1000ull) {
    return 0;
  }
  return (baseline_ns * 1000ull) / candidate_ns;
}

uint64_t div_u64(uint64_t numerator, uint64_t denominator) {
  return denominator == 0 ? 0 : numerator / denominator;
}

void set_cstr(char *dst, size_t dst_len, const char *src) {
  if (dst == nullptr || dst_len == 0) {
    return;
  }
  memset(dst, 0, dst_len);
  if (src == nullptr) {
    return;
  }
  strncpy(dst, src, dst_len - 1);
}

bool path_exists(const char *path) {
  return path != nullptr && access(path, F_OK) == 0;
}

bool shader_compiler_available() {
  return path_exists("/usr/bin/glslc") ||
         path_exists("/usr/bin/glslangValidator") ||
         path_exists("/opt/android-sdk/ndk/27.1.12297006/shader-tools/linux-x86_64/glslc") ||
         path_exists("/opt/android-sdk/ndk/27.0.12077973/shader-tools/linux-x86_64/glslc");
}

bool vulkan_loader_available() {
  return path_exists("/usr/lib/x86_64-linux-gnu/libvulkan.so.1") ||
         path_exists("/usr/local/lib/ollama/vulkan/libvulkan.so.1");
}

void populate_vulkan_rt_availability(
    NervaCudaExperimentalRtCandidateBenchResult *out) {
  out->vulkan_headers_available = NERVA_HAVE_VULKAN_HEADERS ? 1u : 0u;
  out->vulkan_shader_compiler_available =
      shader_compiler_available() ? 1u : 0u;
  out->vulkan_loader_available = vulkan_loader_available() ? 1u : 0u;
#if NERVA_HAVE_VULKAN_HEADERS
  void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (loader == nullptr) {
    return;
  }
  auto vk_get_instance_proc_addr =
      reinterpret_cast<PFN_vkGetInstanceProcAddr>(
          dlsym(loader, "vkGetInstanceProcAddr"));
  if (vk_get_instance_proc_addr == nullptr) {
    dlclose(loader);
    return;
  }
  auto vk_create_instance = reinterpret_cast<PFN_vkCreateInstance>(
      vk_get_instance_proc_addr(nullptr, "vkCreateInstance"));
  if (vk_create_instance == nullptr) {
    dlclose(loader);
    return;
  }

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "nerva-experimental-rt-probe";
  app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.pEngineName = "nerva";
  app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  VkInstance instance = VK_NULL_HANDLE;
  VkResult result = vk_create_instance(&create_info, nullptr, &instance);
  if (result != VK_SUCCESS || instance == VK_NULL_HANDLE) {
    dlclose(loader);
    return;
  }
  auto vk_destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(
      vk_get_instance_proc_addr(instance, "vkDestroyInstance"));
  auto vk_enumerate_physical_devices =
      reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
          vk_get_instance_proc_addr(instance, "vkEnumeratePhysicalDevices"));
  auto vk_enumerate_device_extension_properties =
      reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
          vk_get_instance_proc_addr(instance,
                                    "vkEnumerateDeviceExtensionProperties"));
  if (vk_enumerate_physical_devices != nullptr &&
      vk_enumerate_device_extension_properties != nullptr) {
    uint32_t device_count = 0;
    result = vk_enumerate_physical_devices(instance, &device_count, nullptr);
    if (result == VK_SUCCESS && device_count > 0) {
      out->vulkan_physical_devices = device_count;
      VkPhysicalDevice devices[16];
      uint32_t queried = device_count < 16 ? device_count : 16;
      result = vk_enumerate_physical_devices(instance, &queried, devices);
      for (uint32_t device = 0;
           result == VK_SUCCESS && device < queried &&
           out->vulkan_rt_extensions_available == 0;
           ++device) {
        uint32_t extension_count = 0;
        result = vk_enumerate_device_extension_properties(
            devices[device], nullptr, &extension_count, nullptr);
        if (result != VK_SUCCESS || extension_count == 0) {
          continue;
        }
        VkExtensionProperties extensions[512];
        uint32_t extension_take = extension_count < 512 ? extension_count : 512;
        result = vk_enumerate_device_extension_properties(
            devices[device], nullptr, &extension_take, extensions);
        bool has_accel = false;
        bool has_query = false;
        bool has_pipeline = false;
        for (uint32_t index = 0; result == VK_SUCCESS && index < extension_take;
             ++index) {
          const char *name = extensions[index].extensionName;
          if (strcmp(name, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0) {
            has_accel = true;
          } else if (strcmp(name, VK_KHR_RAY_QUERY_EXTENSION_NAME) == 0) {
            has_query = true;
          } else if (strcmp(name, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) ==
                     0) {
            has_pipeline = true;
          }
        }
        if (has_accel && (has_query || has_pipeline)) {
          out->vulkan_rt_extensions_available = 1;
        }
      }
    }
  }
  if (vk_destroy_instance != nullptr) {
    vk_destroy_instance(instance, nullptr);
  }
  dlclose(loader);
#endif
}

uint32_t ceil_sqrt_u32(uint32_t value) {
  uint32_t root = 1;
  while (static_cast<uint64_t>(root) * root < value) {
    ++root;
  }
  return root;
}

void set_optix_fallback_reason(NervaCudaExperimentalRtCandidateBenchResult *out,
                               const char *stage, const char *detail) {
  char message[192];
  snprintf(message, sizeof(message),
           "OptiX candidate selector unavailable at %s: %s; CUDA fallback used",
           stage == nullptr ? "unknown" : stage,
           detail == nullptr ? "no detail" : detail);
  set_cstr(out->reason, sizeof(out->reason), message);
}

#if NERVA_HAVE_OPTIX_HEADERS

#ifndef NERVA_OPTIX_INCLUDE_DIR
#define NERVA_OPTIX_INCLUDE_DIR ""
#endif

#ifndef NERVA_CUDA_INCLUDE_DIR
#define NERVA_CUDA_INCLUDE_DIR "/usr/local/cuda/include"
#endif

struct RtVertex {
  float x;
  float y;
  float z;
};

struct OptixCandidateParams {
  OptixTraversableHandle handle;
  uint32_t *candidate_pages;
  uint32_t pages;
  uint32_t query_count;
  uint32_t candidates_per_query;
  uint32_t grid_width;
  float cell_size;
};

struct EmptySbtData {
  uint32_t pad;
};

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

using EmptySbtRecord = SbtRecord<EmptySbtData>;

struct OptixCandidateSelector {
  OptixDeviceContext context = nullptr;
  OptixModule module = nullptr;
  OptixPipeline pipeline = nullptr;
  OptixProgramGroup raygen_prog_group = nullptr;
  OptixProgramGroup miss_prog_group = nullptr;
  OptixProgramGroup hitgroup_prog_group = nullptr;
  OptixShaderBindingTable sbt{};
  CUdeviceptr gas_output = 0;
  CUdeviceptr params = 0;
  CUdeviceptr raygen_record = 0;
  CUdeviceptr miss_record = 0;
  CUdeviceptr hitgroup_record = 0;
};

static_assert(sizeof(EmptySbtRecord) % OPTIX_SBT_RECORD_ALIGNMENT == 0,
              "OptiX SBT record size must be aligned");

static const char *kOptixCandidateDeviceSource = R"OPTIX(
#include <optix.h>

struct OptixCandidateParams {
  OptixTraversableHandle handle;
  unsigned int* candidate_pages;
  unsigned int pages;
  unsigned int query_count;
  unsigned int candidates_per_query;
  unsigned int grid_width;
  float cell_size;
};

extern "C" {
__constant__ OptixCandidateParams params;
}

static __forceinline__ __device__ unsigned int hash32(unsigned int value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

extern "C" __global__ void __raygen__candidate() {
  const uint3 idx = optixGetLaunchIndex();
  const unsigned int slot = idx.x;
  const unsigned int query = idx.y;
  if (query >= params.query_count || slot >= params.candidates_per_query ||
      params.pages == 0) {
    return;
  }

  const unsigned int center = hash32(query * 977u + 31u) % params.pages;
  const unsigned int half = params.candidates_per_query / 2u;
  const unsigned int target = (center + params.pages + slot - half) % params.pages;
  const unsigned int x_index = target % params.grid_width;
  const unsigned int y_index = target / params.grid_width;
  const float x = static_cast<float>(x_index) * params.cell_size;
  const float y = static_cast<float>(y_index) * params.cell_size;

  float3 origin;
  origin.x = x;
  origin.y = y;
  origin.z = 1.0f;
  float3 direction;
  direction.x = 0.0f;
  direction.y = 0.0f;
  direction.z = -1.0f;

  unsigned int hit = target;
  optixTrace(params.handle,
             origin,
             direction,
             0.0f,
             2.0f,
             0.0f,
             OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             0,
             1,
             0,
             hit);
  const unsigned long long out_index =
      static_cast<unsigned long long>(query) * params.candidates_per_query + slot;
  params.candidate_pages[out_index] = hit % params.pages;
}

extern "C" __global__ void __miss__candidate() {}

extern "C" __global__ void __closesthit__candidate() {
  optixSetPayload_0(optixGetPrimitiveIndex());
}
)OPTIX";

void optix_log_cb(unsigned int level, const char *tag, const char *message,
                  void *) {
  (void)level;
  (void)tag;
  (void)message;
}

std::string nvrtc_program_log(nvrtcProgram program) {
  size_t log_size = 0;
  nvrtcGetProgramLogSize(program, &log_size);
  if (log_size <= 1) {
    return std::string();
  }
  std::vector<char> log(log_size, 0);
  nvrtcGetProgramLog(program, log.data());
  return std::string(log.data());
}

void set_nvrtc_reason(NervaCudaExperimentalRtCandidateBenchResult *out,
                      const char *stage, nvrtcResult result,
                      const std::string &log = std::string()) {
  char detail[128];
  if (!log.empty()) {
    snprintf(detail, sizeof(detail), "%s: %.96s", nvrtcGetErrorString(result),
             log.c_str());
  } else {
    snprintf(detail, sizeof(detail), "%s", nvrtcGetErrorString(result));
  }
  set_optix_fallback_reason(out, stage, detail);
}

bool compile_optix_candidate_ptx(
    const NervaCudaExperimentalRtCandidateBenchResult *out,
    std::string *ptx, NervaCudaExperimentalRtCandidateBenchResult *result) {
  nvrtcProgram program = nullptr;
  nvrtcResult nvrtc = nvrtcCreateProgram(&program, kOptixCandidateDeviceSource,
                                         "nerva_experimental_rt_optix.cu", 0,
                                         nullptr, nullptr);
  if (nvrtc != NVRTC_SUCCESS) {
    set_nvrtc_reason(result, "nvrtcCreateProgram", nvrtc);
    return false;
  }

  const int major = out->compute_capability_major > 0
                        ? out->compute_capability_major
                        : 7;
  const int minor = out->compute_capability_minor > 0
                        ? out->compute_capability_minor
                        : 0;
  char arch[64];
  snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", major,
           minor);
  std::string optix_include =
      std::string("--include-path=") + NERVA_OPTIX_INCLUDE_DIR;
  std::string cuda_include =
      std::string("--include-path=") + NERVA_CUDA_INCLUDE_DIR;
  const char *options[] = {
      "--std=c++17",
      arch,
      optix_include.c_str(),
      cuda_include.c_str(),
      "--use_fast_math",
  };
  nvrtc = nvrtcCompileProgram(program,
                              static_cast<int>(sizeof(options) / sizeof(options[0])),
                              options);
  if (nvrtc != NVRTC_SUCCESS) {
    std::string log = nvrtc_program_log(program);
    set_nvrtc_reason(result, "nvrtcCompileProgram", nvrtc, log);
    nvrtcDestroyProgram(&program);
    return false;
  }

  size_t ptx_size = 0;
  nvrtc = nvrtcGetPTXSize(program, &ptx_size);
  if (nvrtc != NVRTC_SUCCESS || ptx_size == 0) {
    set_nvrtc_reason(result, "nvrtcGetPTXSize", nvrtc);
    nvrtcDestroyProgram(&program);
    return false;
  }
  ptx->assign(ptx_size, '\0');
  nvrtc = nvrtcGetPTX(program, ptx->data());
  nvrtcDestroyProgram(&program);
  if (nvrtc != NVRTC_SUCCESS) {
    set_nvrtc_reason(result, "nvrtcGetPTX", nvrtc);
    return false;
  }
  return true;
}

void cleanup_optix_selector(OptixCandidateSelector *state,
                            NervaCudaExperimentalRtCandidateBenchResult *out) {
  if (state->hitgroup_record != 0) {
    cudaFree(reinterpret_cast<void *>(state->hitgroup_record));
    out->device_frees += 1;
    state->hitgroup_record = 0;
  }
  if (state->miss_record != 0) {
    cudaFree(reinterpret_cast<void *>(state->miss_record));
    out->device_frees += 1;
    state->miss_record = 0;
  }
  if (state->raygen_record != 0) {
    cudaFree(reinterpret_cast<void *>(state->raygen_record));
    out->device_frees += 1;
    state->raygen_record = 0;
  }
  if (state->params != 0) {
    cudaFree(reinterpret_cast<void *>(state->params));
    out->device_frees += 1;
    state->params = 0;
  }
  if (state->gas_output != 0) {
    cudaFree(reinterpret_cast<void *>(state->gas_output));
    out->device_frees += 1;
    state->gas_output = 0;
  }
  if (state->pipeline != nullptr) {
    optixPipelineDestroy(state->pipeline);
    state->pipeline = nullptr;
  }
  if (state->hitgroup_prog_group != nullptr) {
    optixProgramGroupDestroy(state->hitgroup_prog_group);
    state->hitgroup_prog_group = nullptr;
  }
  if (state->miss_prog_group != nullptr) {
    optixProgramGroupDestroy(state->miss_prog_group);
    state->miss_prog_group = nullptr;
  }
  if (state->raygen_prog_group != nullptr) {
    optixProgramGroupDestroy(state->raygen_prog_group);
    state->raygen_prog_group = nullptr;
  }
  if (state->module != nullptr) {
    optixModuleDestroy(state->module);
    state->module = nullptr;
  }
  if (state->context != nullptr) {
    optixDeviceContextDestroy(state->context);
    state->context = nullptr;
  }
}

bool optix_ok(NervaCudaExperimentalRtCandidateBenchResult *out,
              const char *stage, OptixResult result, const char *log = nullptr) {
  if (result == OPTIX_SUCCESS) {
    return true;
  }
  char detail[128];
  if (log != nullptr && log[0] != '\0') {
    snprintf(detail, sizeof(detail), "OptixResult %d: %.96s",
             static_cast<int>(result), log);
  } else {
    snprintf(detail, sizeof(detail), "OptixResult %d",
             static_cast<int>(result));
  }
  set_optix_fallback_reason(out, stage, detail);
  return false;
}

bool pack_empty_sbt_record(OptixProgramGroup group, CUdeviceptr *device_record,
                           NervaCudaExperimentalRtCandidateBenchResult *out,
                           const char *stage) {
  EmptySbtRecord record{};
  OptixResult optix = optixSbtRecordPackHeader(group, &record);
  if (!optix_ok(out, stage, optix)) {
    return false;
  }
  cudaError_t err =
      cudaMalloc(reinterpret_cast<void **>(device_record), sizeof(record));
  if (err != cudaSuccess) {
    return false;
  }
  out->device_allocations += 1;
  out->device_arena_bytes += sizeof(record);
  err = cudaMemcpy(reinterpret_cast<void *>(*device_record), &record,
                   sizeof(record), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    set_optix_fallback_reason(out, stage, cudaGetErrorString(err));
    return false;
  }
  return true;
}

bool create_optix_candidate_selector(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    uint32_t *candidate_pages, cudaStream_t stream,
    OptixCandidateSelector *state,
    NervaCudaExperimentalRtCandidateBenchResult *out) {
  if (out->rt_core_capable == 0) {
    set_optix_fallback_reason(out, "device capability",
                              "compute capability is below RTX-era GPUs");
    return false;
  }

  cudaError_t err = cudaFree(nullptr);
  if (err != cudaSuccess) {
    set_optix_fallback_reason(out, "cudaFree(0)", cudaGetErrorString(err));
    return false;
  }

  OptixResult optix = optixInit();
  if (!optix_ok(out, "optixInit", optix)) {
    return false;
  }

  OptixDeviceContextOptions context_options{};
  context_options.logCallbackFunction = &optix_log_cb;
  context_options.logCallbackLevel = 1;
  optix = optixDeviceContextCreate(0, &context_options, &state->context);
  if (!optix_ok(out, "optixDeviceContextCreate", optix)) {
    return false;
  }

  const uint32_t grid_width = ceil_sqrt_u32(request->pages);
  constexpr float cell_size = 2.0f;
  constexpr float half_size = 0.45f;
  std::vector<RtVertex> vertices(static_cast<size_t>(request->pages) * 3u);
  for (uint32_t page = 0; page < request->pages; ++page) {
    const float x = static_cast<float>(page % grid_width) * cell_size;
    const float y = static_cast<float>(page / grid_width) * cell_size;
    vertices[static_cast<size_t>(page) * 3u + 0u] =
        RtVertex{x - half_size, y - half_size, 0.0f};
    vertices[static_cast<size_t>(page) * 3u + 1u] =
        RtVertex{x + half_size, y - half_size, 0.0f};
    vertices[static_cast<size_t>(page) * 3u + 2u] =
        RtVertex{x, y + half_size, 0.0f};
  }

  CUdeviceptr d_vertices = 0;
  CUdeviceptr d_temp = 0;
  auto cleanup_build_buffers = [&]() {
    if (d_temp != 0) {
      cudaFree(reinterpret_cast<void *>(d_temp));
      out->device_frees += 1;
      d_temp = 0;
    }
    if (d_vertices != 0) {
      cudaFree(reinterpret_cast<void *>(d_vertices));
      out->device_frees += 1;
      d_vertices = 0;
    }
  };

  const size_t vertex_bytes = vertices.size() * sizeof(RtVertex);
  err = cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertex_bytes);
  if (err != cudaSuccess) {
    set_optix_fallback_reason(out, "cudaMalloc(vertices)",
                              cudaGetErrorString(err));
    return false;
  }
  out->device_allocations += 1;
  out->device_arena_bytes += vertex_bytes;
  err = cudaMemcpyAsync(reinterpret_cast<void *>(d_vertices), vertices.data(),
                        vertex_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    cleanup_build_buffers();
    set_optix_fallback_reason(out, "cudaMemcpy(vertices)",
                              cudaGetErrorString(err));
    return false;
  }

  const uint32_t triangle_flags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
  OptixBuildInput triangle_input{};
  triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangle_input.triangleArray.vertexStrideInBytes = sizeof(RtVertex);
  triangle_input.triangleArray.numVertices =
      static_cast<uint32_t>(vertices.size());
  triangle_input.triangleArray.vertexBuffers = &d_vertices;
  triangle_input.triangleArray.flags = triangle_flags;
  triangle_input.triangleArray.numSbtRecords = 1;

  OptixAccelBuildOptions accel_options{};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_sizes{};
  optix = optixAccelComputeMemoryUsage(state->context, &accel_options,
                                       &triangle_input, 1, &gas_sizes);
  if (!optix_ok(out, "optixAccelComputeMemoryUsage", optix)) {
    cleanup_build_buffers();
    return false;
  }

  err = cudaMalloc(reinterpret_cast<void **>(&d_temp),
                   gas_sizes.tempSizeInBytes);
  if (err != cudaSuccess) {
    cleanup_build_buffers();
    set_optix_fallback_reason(out, "cudaMalloc(gas temp)",
                              cudaGetErrorString(err));
    return false;
  }
  out->device_allocations += 1;
  out->device_arena_bytes += gas_sizes.tempSizeInBytes;
  err = cudaMalloc(reinterpret_cast<void **>(&state->gas_output),
                   gas_sizes.outputSizeInBytes);
  if (err != cudaSuccess) {
    cleanup_build_buffers();
    set_optix_fallback_reason(out, "cudaMalloc(gas output)",
                              cudaGetErrorString(err));
    return false;
  }
  out->device_allocations += 1;
  out->device_arena_bytes += gas_sizes.outputSizeInBytes;

  OptixTraversableHandle gas_handle = 0;
  optix = optixAccelBuild(state->context, reinterpret_cast<CUstream>(stream),
                          &accel_options, &triangle_input, 1, d_temp,
                          gas_sizes.tempSizeInBytes, state->gas_output,
                          gas_sizes.outputSizeInBytes, &gas_handle, nullptr, 0);
  cleanup_build_buffers();
  if (!optix_ok(out, "optixAccelBuild", optix)) {
    return false;
  }

  std::string ptx;
  if (!compile_optix_candidate_ptx(out, &ptx, out)) {
    return false;
  }

  OptixModuleCompileOptions module_options{};
  module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

  OptixPipelineCompileOptions pipeline_options{};
  pipeline_options.usesMotionBlur = false;
  pipeline_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_options.numPayloadValues = 1;
  pipeline_options.numAttributeValues = 2;
  pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_options.pipelineLaunchParamsVariableName = "params";
  pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  char log[2048];
  size_t log_size = sizeof(log);
  optix = optixModuleCreate(state->context, &module_options, &pipeline_options,
                            ptx.c_str(), ptx.size(), log, &log_size,
                            &state->module);
  if (!optix_ok(out, "optixModuleCreate", optix, log)) {
    return false;
  }

  OptixProgramGroupOptions group_options{};
  OptixProgramGroupDesc raygen_desc{};
  raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygen_desc.raygen.module = state->module;
  raygen_desc.raygen.entryFunctionName = "__raygen__candidate";
  log_size = sizeof(log);
  optix = optixProgramGroupCreate(state->context, &raygen_desc, 1,
                                  &group_options, log, &log_size,
                                  &state->raygen_prog_group);
  if (!optix_ok(out, "optixProgramGroupCreate(raygen)", optix, log)) {
    return false;
  }

  OptixProgramGroupDesc miss_desc{};
  miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  miss_desc.miss.module = state->module;
  miss_desc.miss.entryFunctionName = "__miss__candidate";
  log_size = sizeof(log);
  optix = optixProgramGroupCreate(state->context, &miss_desc, 1,
                                  &group_options, log, &log_size,
                                  &state->miss_prog_group);
  if (!optix_ok(out, "optixProgramGroupCreate(miss)", optix, log)) {
    return false;
  }

  OptixProgramGroupDesc hit_desc{};
  hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hit_desc.hitgroup.moduleCH = state->module;
  hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__candidate";
  log_size = sizeof(log);
  optix = optixProgramGroupCreate(state->context, &hit_desc, 1, &group_options,
                                  log, &log_size,
                                  &state->hitgroup_prog_group);
  if (!optix_ok(out, "optixProgramGroupCreate(hitgroup)", optix, log)) {
    return false;
  }

  OptixProgramGroup groups[3] = {state->raygen_prog_group,
                                 state->miss_prog_group,
                                 state->hitgroup_prog_group};
  OptixPipelineLinkOptions link_options{};
  link_options.maxTraceDepth = 1;
  log_size = sizeof(log);
  optix = optixPipelineCreate(state->context, &pipeline_options, &link_options,
                              groups, 3, log, &log_size, &state->pipeline);
  if (!optix_ok(out, "optixPipelineCreate", optix, log)) {
    return false;
  }

  OptixStackSizes stack_sizes{};
  for (OptixProgramGroup group : groups) {
    optix = optixUtilAccumulateStackSizes(group, &stack_sizes, state->pipeline);
    if (!optix_ok(out, "optixUtilAccumulateStackSizes", optix)) {
      return false;
    }
  }
  uint32_t direct_callable_stack_from_traversal = 0;
  uint32_t direct_callable_stack_from_state = 0;
  uint32_t continuation_stack = 0;
  optix = optixUtilComputeStackSizes(&stack_sizes, 1, 0, 0,
                                     &direct_callable_stack_from_traversal,
                                     &direct_callable_stack_from_state,
                                     &continuation_stack);
  if (!optix_ok(out, "optixUtilComputeStackSizes", optix)) {
    return false;
  }
  optix = optixPipelineSetStackSize(state->pipeline,
                                    direct_callable_stack_from_traversal,
                                    direct_callable_stack_from_state,
                                    continuation_stack, 1);
  if (!optix_ok(out, "optixPipelineSetStackSize", optix)) {
    return false;
  }

  if (!pack_empty_sbt_record(state->raygen_prog_group, &state->raygen_record,
                             out, "pack raygen SBT") ||
      !pack_empty_sbt_record(state->miss_prog_group, &state->miss_record, out,
                             "pack miss SBT") ||
      !pack_empty_sbt_record(state->hitgroup_prog_group, &state->hitgroup_record,
                             out, "pack hitgroup SBT")) {
    return false;
  }
  state->sbt.raygenRecord = state->raygen_record;
  state->sbt.missRecordBase = state->miss_record;
  state->sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
  state->sbt.missRecordCount = 1;
  state->sbt.hitgroupRecordBase = state->hitgroup_record;
  state->sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
  state->sbt.hitgroupRecordCount = 1;

  OptixCandidateParams params{};
  params.handle = gas_handle;
  params.candidate_pages = candidate_pages;
  params.pages = request->pages;
  params.query_count = request->query_count;
  params.candidates_per_query = request->candidates_per_query;
  params.grid_width = grid_width;
  params.cell_size = cell_size;
  err = cudaMalloc(reinterpret_cast<void **>(&state->params), sizeof(params));
  if (err != cudaSuccess) {
    set_optix_fallback_reason(out, "cudaMalloc(params)",
                              cudaGetErrorString(err));
    return false;
  }
  out->device_allocations += 1;
  out->device_arena_bytes += sizeof(params);
  err = cudaMemcpyAsync(reinterpret_cast<void *>(state->params), &params,
                        sizeof(params), cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    set_optix_fallback_reason(out, "cudaMemcpy(params)",
                              cudaGetErrorString(err));
    return false;
  }
  return true;
}

bool time_optix_selector(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    OptixCandidateSelector *state, cudaStream_t stream, cudaEvent_t start,
    cudaEvent_t stop, uint64_t *elapsed,
    NervaCudaExperimentalRtCandidateBenchResult *out) {
  for (uint32_t iter = 0; iter < request->warmup_iterations; ++iter) {
    OptixResult optix =
        optixLaunch(state->pipeline, reinterpret_cast<CUstream>(stream),
                    state->params, sizeof(OptixCandidateParams), &state->sbt,
                    request->candidates_per_query, request->query_count, 1);
    if (!optix_ok(out, "optixLaunch(warmup)", optix)) {
      return false;
    }
  }
  cudaError_t err = cudaEventRecord(start, stream);
  if (err != cudaSuccess) {
    set_optix_fallback_reason(out, "cudaEventRecord(start)",
                              cudaGetErrorString(err));
    return false;
  }
  for (uint32_t iter = 0; iter < request->iterations; ++iter) {
    OptixResult optix =
        optixLaunch(state->pipeline, reinterpret_cast<CUstream>(stream),
                    state->params, sizeof(OptixCandidateParams), &state->sbt,
                    request->candidates_per_query, request->query_count, 1);
    if (!optix_ok(out, "optixLaunch", optix)) {
      return false;
    }
  }
  err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) {
    err = cudaEventSynchronize(stop);
  }
  if (err != cudaSuccess) {
    set_optix_fallback_reason(out, "cudaEventSynchronize(stop)",
                              cudaGetErrorString(err));
    return false;
  }
  *elapsed = elapsed_ns(start, stop);
  return true;
}

#endif

__device__ uint32_t hash32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

__device__ float deterministic_f32(uint32_t a, uint32_t b) {
  const uint32_t bits = hash32(a * 1315423911u + b * 2654435761u);
  const int32_t centered = static_cast<int32_t>(bits % 4096u) - 2048;
  return static_cast<float>(centered) * (1.0f / 2048.0f);
}

__global__ void init_page_descriptors_kernel(float *descriptors, uint32_t pages,
                                             uint32_t dims) {
  const uint64_t total = static_cast<uint64_t>(pages) * dims;
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  while (index < total) {
    const uint32_t page = static_cast<uint32_t>(index / dims);
    const uint32_t dim = static_cast<uint32_t>(index - static_cast<uint64_t>(page) * dims);
    descriptors[index] = deterministic_f32(page, dim);
    index += stride;
  }
}

__global__ void init_query_descriptors_kernel(float *queries, uint32_t query_count,
                                              uint32_t dims) {
  const uint64_t total = static_cast<uint64_t>(query_count) * dims;
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  while (index < total) {
    const uint32_t query = static_cast<uint32_t>(index / dims);
    const uint32_t dim = static_cast<uint32_t>(index - static_cast<uint64_t>(query) * dims);
    queries[index] = deterministic_f32(query + 0x9e37u, dim + 17u);
    index += stride;
  }
}

__device__ float descriptor_dot(const float *descriptors, const float *queries,
                                uint32_t page, uint32_t query, uint32_t dims) {
  float sum = 0.0f;
  const uint64_t page_base = static_cast<uint64_t>(page) * dims;
  const uint64_t query_base = static_cast<uint64_t>(query) * dims;
  for (uint32_t dim = 0; dim < dims; ++dim) {
    sum += descriptors[page_base + dim] * queries[query_base + dim];
  }
  return sum;
}

__global__ void dense_selector_kernel(const float *descriptors, const float *queries,
                                      uint32_t pages, uint32_t dims,
                                      uint32_t query_count, uint32_t *out_pages) {
  const uint32_t query = blockIdx.x;
  if (query >= query_count) {
    return;
  }
  float best_score = -INFINITY;
  uint32_t best_page = 0;
  for (uint32_t page = threadIdx.x; page < pages; page += blockDim.x) {
    const float score = descriptor_dot(descriptors, queries, page, query, dims);
    if (score > best_score) {
      best_score = score;
      best_page = page;
    }
  }
  __shared__ float scores[kThreads];
  __shared__ uint32_t pages_shared[kThreads];
  scores[threadIdx.x] = best_score;
  pages_shared[threadIdx.x] = best_page;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_score = scores[threadIdx.x + stride];
      const uint32_t other_page = pages_shared[threadIdx.x + stride];
      if (other_score > scores[threadIdx.x] ||
          (other_score == scores[threadIdx.x] && other_page < pages_shared[threadIdx.x])) {
        scores[threadIdx.x] = other_score;
        pages_shared[threadIdx.x] = other_page;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out_pages[query] = pages_shared[0];
  }
}

__global__ void software_candidate_selector_kernel(uint32_t *candidate_pages,
                                                   uint32_t pages,
                                                   uint32_t query_count,
                                                   uint32_t candidates_per_query) {
  const uint32_t query = blockIdx.x;
  if (query >= query_count || pages == 0) {
    return;
  }
  const uint32_t center = hash32(query * 977u + 31u) % pages;
  const uint32_t half = candidates_per_query / 2u;
  for (uint32_t offset = threadIdx.x; offset < candidates_per_query; offset += blockDim.x) {
    const uint32_t wrapped = center + pages + offset - half;
    candidate_pages[static_cast<uint64_t>(query) * candidates_per_query + offset] =
        wrapped % pages;
  }
}

__global__ void rerank_candidate_kernel(const float *descriptors, const float *queries,
                                        const uint32_t *candidate_pages, uint32_t pages,
                                        uint32_t dims, uint32_t query_count,
                                        uint32_t candidates_per_query,
                                        uint32_t *out_pages) {
  const uint32_t query = blockIdx.x;
  if (query >= query_count || pages == 0) {
    return;
  }
  float best_score = -INFINITY;
  uint32_t best_page = 0;
  for (uint32_t index = threadIdx.x; index < candidates_per_query; index += blockDim.x) {
    const uint32_t page =
        candidate_pages[static_cast<uint64_t>(query) * candidates_per_query + index] % pages;
    const float score = descriptor_dot(descriptors, queries, page, query, dims);
    if (score > best_score) {
      best_score = score;
      best_page = page;
    }
  }
  __shared__ float scores[kThreads];
  __shared__ uint32_t pages_shared[kThreads];
  scores[threadIdx.x] = best_score;
  pages_shared[threadIdx.x] = best_page;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_score = scores[threadIdx.x + stride];
      const uint32_t other_page = pages_shared[threadIdx.x + stride];
      if (other_score > scores[threadIdx.x] ||
          (other_score == scores[threadIdx.x] && other_page < pages_shared[threadIdx.x])) {
        scores[threadIdx.x] = other_score;
        pages_shared[threadIdx.x] = other_page;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out_pages[query] = pages_shared[0];
  }
}

cudaError_t time_dense_selector(const NervaCudaExperimentalRtCandidateBenchRequest *request,
                                const float *descriptors, const float *queries,
                                uint32_t *out_pages, cudaStream_t stream,
                                cudaEvent_t start, cudaEvent_t stop,
                                uint64_t *elapsed) {
  for (uint32_t iter = 0; iter < request->warmup_iterations; ++iter) {
    dense_selector_kernel<<<request->query_count, kThreads, 0, stream>>>(
        descriptors, queries, request->pages, request->dims, request->query_count, out_pages);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaEventRecord(start, stream);
  if (err != cudaSuccess) {
    return err;
  }
  for (uint32_t iter = 0; iter < request->iterations; ++iter) {
    dense_selector_kernel<<<request->query_count, kThreads, 0, stream>>>(
        descriptors, queries, request->pages, request->dims, request->query_count, out_pages);
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) {
    err = cudaEventSynchronize(stop);
  }
  if (err != cudaSuccess) {
    return err;
  }
  *elapsed = elapsed_ns(start, stop);
  return cudaSuccess;
}

cudaError_t time_software_selector(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    uint32_t *candidate_pages, cudaStream_t stream, cudaEvent_t start,
    cudaEvent_t stop, uint64_t *elapsed) {
  for (uint32_t iter = 0; iter < request->warmup_iterations; ++iter) {
    software_candidate_selector_kernel<<<request->query_count, kThreads, 0, stream>>>(
        candidate_pages, request->pages, request->query_count, request->candidates_per_query);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaEventRecord(start, stream);
  if (err != cudaSuccess) {
    return err;
  }
  for (uint32_t iter = 0; iter < request->iterations; ++iter) {
    software_candidate_selector_kernel<<<request->query_count, kThreads, 0, stream>>>(
        candidate_pages, request->pages, request->query_count, request->candidates_per_query);
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) {
    err = cudaEventSynchronize(stop);
  }
  if (err != cudaSuccess) {
    return err;
  }
  *elapsed = elapsed_ns(start, stop);
  return cudaSuccess;
}

cudaError_t time_rerank(const NervaCudaExperimentalRtCandidateBenchRequest *request,
                        const float *descriptors, const float *queries,
                        const uint32_t *candidate_pages, uint32_t *out_pages,
                        cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop,
                        uint64_t *elapsed) {
  for (uint32_t iter = 0; iter < request->warmup_iterations; ++iter) {
    rerank_candidate_kernel<<<request->query_count, kThreads, 0, stream>>>(
        descriptors, queries, candidate_pages, request->pages, request->dims,
        request->query_count, request->candidates_per_query, out_pages);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaEventRecord(start, stream);
  if (err != cudaSuccess) {
    return err;
  }
  for (uint32_t iter = 0; iter < request->iterations; ++iter) {
    rerank_candidate_kernel<<<request->query_count, kThreads, 0, stream>>>(
        descriptors, queries, candidate_pages, request->pages, request->dims,
        request->query_count, request->candidates_per_query, out_pages);
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) {
    err = cudaEventSynchronize(stop);
  }
  if (err != cudaSuccess) {
    return err;
  }
  *elapsed = elapsed_ns(start, stop);
  return cudaSuccess;
}

uint64_t hash_selected_pages(const uint32_t *pages, uint32_t count) {
  uint64_t hash = 1469598103934665603ull;
  for (uint32_t index = 0; index < count; ++index) {
    hash ^= pages[index];
    hash *= 1099511628211ull;
  }
  return hash;
}

void clear_result(const NervaCudaExperimentalRtCandidateBenchRequest *request,
                  NervaCudaExperimentalRtCandidateBenchResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  set_cstr(out->backend, sizeof(out->backend), "software_cuda_candidate_selector");
  set_cstr(out->reason, sizeof(out->reason),
           "CUDA fallback: OptiX/Vulkan RT SDK headers are not part of this build; "
           "these numbers do not use RT cores");
  if (request == nullptr) {
    return;
  }
  out->pages = request->pages;
  out->page_tokens = request->page_tokens;
  out->dims = request->dims;
  out->query_count = request->query_count;
  out->candidates_per_query = request->candidates_per_query;
  out->iterations = request->iterations;
  out->warmup_iterations = request->warmup_iterations;
  out->descriptor_bytes =
      checked_mul_u64(checked_mul_u64(request->pages, request->dims), sizeof(float));
  out->query_bytes =
      checked_mul_u64(checked_mul_u64(request->query_count, request->dims), sizeof(float));
  out->candidate_id_bytes = checked_mul_u64(
      checked_mul_u64(request->query_count, request->candidates_per_query), sizeof(uint32_t));
  out->output_bytes = checked_mul_u64(request->query_count, sizeof(uint32_t)) * 2ull;
  out->device_arena_bytes = out->descriptor_bytes + out->query_bytes +
                            out->candidate_id_bytes + out->output_bytes;
}

int fail(NervaCudaExperimentalRtCandidateBenchResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

}  // namespace

extern "C" int nerva_cuda_experimental_rt_candidate_bench(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    NervaCudaExperimentalRtCandidateBenchResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(request, out);
  if (request == nullptr || request->pages == 0 || request->page_tokens == 0 ||
      request->dims == 0 || request->dims > 256 || request->query_count == 0 ||
      request->candidates_per_query == 0 ||
      request->candidates_per_query > request->pages ||
      request->iterations == 0) {
    return fail(out, cudaErrorInvalidValue);
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    return fail(out, cudaErrorNoDevice);
  }
  err = cudaGetDevice(&out->device_ordinal);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  cudaDeviceProp props{};
  err = cudaGetDeviceProperties(&props, out->device_ordinal);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  out->compute_capability_major = props.major;
  out->compute_capability_minor = props.minor;
  out->rt_core_capable = props.major >= 7 ? 1u : 0u;
  out->optix_headers_available = NERVA_HAVE_OPTIX_HEADERS ? 1u : 0u;
  populate_vulkan_rt_availability(out);
  out->rt_headers_available =
      (out->optix_headers_available != 0 || out->vulkan_headers_available != 0)
          ? 1u
          : 0u;
  out->real_rt_backend_available = 0;
  if (out->optix_headers_available == 0 &&
      out->vulkan_rt_extensions_available != 0) {
    set_cstr(out->reason, sizeof(out->reason),
             "OptiX SDK headers are not installed; Vulkan RT appears "
             "available, but this bench currently uses the CUDA fallback "
             "selector and does not use RT cores");
  } else if (out->optix_headers_available != 0) {
    set_cstr(out->reason, sizeof(out->reason),
             "OptiX SDK headers are installed; attempting OptiX candidate "
             "selector backend");
  }

  float *descriptors = nullptr;
  float *queries = nullptr;
  uint32_t *candidate_pages = nullptr;
  uint32_t *dense_out = nullptr;
  uint32_t *candidate_out = nullptr;
  uint32_t *host_selected = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  auto cleanup = [&]() {
    if (host_selected != nullptr) {
      delete[] host_selected;
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    if (candidate_out != nullptr) {
      cudaFree(candidate_out);
      out->device_frees += 1;
    }
    if (dense_out != nullptr) {
      cudaFree(dense_out);
      out->device_frees += 1;
    }
    if (candidate_pages != nullptr) {
      cudaFree(candidate_pages);
      out->device_frees += 1;
    }
    if (queries != nullptr) {
      cudaFree(queries);
      out->device_frees += 1;
    }
    if (descriptors != nullptr) {
      cudaFree(descriptors);
      out->device_frees += 1;
    }
  };
  auto fail_with_cleanup = [&](cudaError_t failure) {
    cleanup();
    return fail(out, failure);
  };

  err = cudaMalloc(reinterpret_cast<void **>(&descriptors), out->descriptor_bytes);
  if (err == cudaSuccess) out->device_allocations += 1;
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&queries), out->query_bytes);
    if (err == cudaSuccess) out->device_allocations += 1;
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&candidate_pages), out->candidate_id_bytes);
    if (err == cudaSuccess) out->device_allocations += 1;
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&dense_out),
                     request->query_count * sizeof(uint32_t));
    if (err == cudaSuccess) out->device_allocations += 1;
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&candidate_out),
                     request->query_count * sizeof(uint32_t));
    if (err == cudaSuccess) out->device_allocations += 1;
  }
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  host_selected = new (std::nothrow) uint32_t[request->query_count];
  if (host_selected == nullptr) {
    return fail_with_cleanup(cudaErrorMemoryAllocation);
  }

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err == cudaSuccess) {
    err = cudaEventCreate(&start);
  }
  if (err == cudaSuccess) {
    err = cudaEventCreate(&stop);
  }
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }

  const uint64_t descriptor_elements =
      static_cast<uint64_t>(request->pages) * request->dims;
  const uint64_t query_elements =
      static_cast<uint64_t>(request->query_count) * request->dims;
  const uint32_t descriptor_blocks = static_cast<uint32_t>(
      descriptor_elements / kThreads + (descriptor_elements % kThreads != 0));
  const uint32_t query_blocks = static_cast<uint32_t>(
      query_elements / kThreads + (query_elements % kThreads != 0));
  init_page_descriptors_kernel<<<descriptor_blocks > kMaxInitBlocks ? kMaxInitBlocks : descriptor_blocks,
                                 kThreads, 0, stream>>>(descriptors, request->pages, request->dims);
  init_query_descriptors_kernel<<<query_blocks > kMaxInitBlocks ? kMaxInitBlocks : query_blocks,
                                  kThreads, 0, stream>>>(queries, request->query_count,
                                                         request->dims);
  err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->kernel_launches += 2;
  out->sync_calls += 1;

  err = time_dense_selector(request, descriptors, queries, dense_out, stream, start, stop,
                            &out->dense_selector_total_ns);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->sync_calls += 1;
  out->kernel_launches += request->warmup_iterations + request->iterations;

  bool used_optix_selector = false;
#if NERVA_HAVE_OPTIX_HEADERS
  if (out->optix_headers_available != 0) {
    OptixCandidateSelector optix_selector{};
    if (create_optix_candidate_selector(request, candidate_pages, stream,
                                        &optix_selector, out)) {
      used_optix_selector =
          time_optix_selector(request, &optix_selector, stream, start, stop,
                              &out->software_selector_total_ns, out);
      if (used_optix_selector) {
        set_cstr(out->backend, sizeof(out->backend),
                 "optix_rt_candidate_selector");
        set_cstr(out->reason, sizeof(out->reason),
                 "OptiX hardware traversal generated candidate IDs; exact "
                 "rerank remains CUDA");
        out->real_rt_backend_available = 1;
      }
    }
    cleanup_optix_selector(&optix_selector, out);
  }
#endif
  if (!used_optix_selector) {
    err = time_software_selector(request, candidate_pages, stream, start, stop,
                                 &out->software_selector_total_ns);
    if (err != cudaSuccess) {
      return fail_with_cleanup(err);
    }
  }
  out->sync_calls += 1;
  out->kernel_launches += request->warmup_iterations + request->iterations;

  err = time_rerank(request, descriptors, queries, candidate_pages, candidate_out, stream,
                    start, stop, &out->rerank_total_ns);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->sync_calls += 1;
  out->kernel_launches += request->warmup_iterations + request->iterations;

  err = cudaMemcpyAsync(host_selected, candidate_out,
                        request->query_count * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream);
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->sync_calls += 1;
  out->selected_hash = hash_selected_pages(host_selected, request->query_count);
  out->dense_selector_avg_ns = out->dense_selector_total_ns / request->iterations;
  out->software_selector_avg_ns =
      out->software_selector_total_ns / request->iterations;
  out->rerank_avg_ns = out->rerank_total_ns / request->iterations;
  out->selector_plus_rerank_avg_ns =
      out->software_selector_avg_ns + out->rerank_avg_ns;
  out->dense_vs_selector_speedup_x1000 =
      speedup_x1000(out->dense_selector_avg_ns, out->software_selector_avg_ns);
  out->dense_vs_selector_plus_rerank_speedup_x1000 =
      speedup_x1000(out->dense_selector_avg_ns, out->selector_plus_rerank_avg_ns);
  out->candidate_fraction_ppm =
      div_u64(static_cast<uint64_t>(request->candidates_per_query) * 1000000ull,
              request->pages);
  out->hot_path_allocations = 0;
  out->status = 0;

  cleanup();
  return 0;
}
