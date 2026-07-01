#include "nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <math.h>
#include <new>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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
constexpr uint32_t kMaxAttentionDims = 32;
constexpr uint32_t kMaxOracleTopK = 128;
constexpr uint32_t kDefaultLocalWindowTokens = 8192;

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

uint64_t mul_div_u64(uint64_t lhs, uint64_t rhs, uint64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  const __uint128_t product =
      static_cast<__uint128_t>(lhs) * static_cast<__uint128_t>(rhs);
  const __uint128_t value = product / divisor;
  return value > UINT64_MAX ? UINT64_MAX : static_cast<uint64_t>(value);
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
         path_exists("/opt/android-sdk/ndk/27.1.12297006/shader-tools/linux-aarch64/glslc") ||
         path_exists("/opt/android-sdk/ndk/27.0.12077973/shader-tools/linux-x86_64/glslc") ||
         path_exists("/opt/android-sdk/ndk/27.0.12077973/shader-tools/linux-aarch64/glslc");
}

bool vulkan_loader_available() {
  return path_exists("/usr/lib/x86_64-linux-gnu/libvulkan.so.1") ||
         path_exists("/usr/lib/aarch64-linux-gnu/libvulkan.so.1") ||
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

bool env_truthy(const char *name) {
  const char *value = getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
         value[0] == 't' || value[0] == 'T';
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
  uint32_t layer_index;
  uint32_t layer_count;
  uint32_t grid_width;
  uint32_t current_page;
  uint32_t local_pages;
  uint32_t sink_pages;
  const uint32_t *step_cursor;
  uint32_t page_tokens_for_step;
  uint32_t dynamic_step;
  float cell_size;
  const float *queries;
  uint32_t query_dims;
  uint32_t query_derived_pages;
  uint32_t descriptor_geometry;
  float descriptor_scale;
  float descriptor_plane_stride;
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
  OptixTraversableHandle traversable = 0;
  const float *queries = nullptr;
  const uint32_t *step_cursor = nullptr;
  uint32_t query_dims = 0;
  uint32_t query_derived_pages = 0;
  uint32_t descriptor_geometry = 0;
  uint32_t layer_count = 1;
  uint32_t dynamic_step = 0;
  uint32_t grid_width = 0;
  float cell_size = 0.0f;
  float descriptor_scale = 16.0f;
  float descriptor_plane_stride = 4.0f;
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
  unsigned int layer_index;
  unsigned int layer_count;
  unsigned int grid_width;
  unsigned int current_page;
  unsigned int local_pages;
  unsigned int sink_pages;
  const unsigned int* step_cursor;
  unsigned int page_tokens_for_step;
  unsigned int dynamic_step;
  float cell_size;
  const float* queries;
  unsigned int query_dims;
  unsigned int query_derived_pages;
  unsigned int descriptor_geometry;
  float descriptor_scale;
  float descriptor_plane_stride;
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

static __forceinline__ __device__ unsigned int query_center_page(
    unsigned int query, unsigned int pages, unsigned int candidates_per_query) {
  if (pages == 0u) {
    return 0u;
  }
  const unsigned int seed = hash32(query * 977u + 31u);
  if (candidates_per_query > 0u && candidates_per_query < pages) {
    const unsigned int half = candidates_per_query / 2u;
    const unsigned int span = pages - candidates_per_query + 1u;
    return half + (seed % span);
  }
  return seed % pages;
}

static __forceinline__ __device__ unsigned int query_descriptor_center_page(
    unsigned int query, unsigned int pages) {
  if (pages == 0u || params.queries == 0 || params.query_dims < 2u ||
      params.query_derived_pages == 0u) {
    return pages == 0u ? 0u : query % pages;
  }
  const unsigned long long base =
      static_cast<unsigned long long>(query) * params.query_dims;
  const float q0 = params.queries[base];
  const float q1 = params.queries[base + 1ull];
  if (!(q1 < -1.0e-20f || q1 > 1.0e-20f)) {
    return query % pages;
  }
  const float center_position = -0.5f * q0 / q1;
  float normalized = center_position;
  normalized = fminf(1.0f, fmaxf(0.0f, normalized));
  const float scaled = normalized * static_cast<float>(pages - 1u);
  unsigned int page = static_cast<unsigned int>(floorf(scaled + 0.5f));
  return page < pages ? page : pages - 1u;
}

static __forceinline__ __device__ unsigned int min_u32(unsigned int a,
                                                       unsigned int b) {
  return a < b ? a : b;
}

extern "C" __global__ void __raygen__candidate() {
  const uint3 idx = optixGetLaunchIndex();
  const unsigned int slot = idx.x;
  const unsigned int query = idx.y;
  if (query >= params.query_count || slot >= params.candidates_per_query ||
      params.pages == 0) {
    return;
  }

  unsigned int pages = params.pages;
  unsigned int current_page =
      params.current_page < pages ? params.current_page : pages - 1u;
  if (params.dynamic_step != 0u && params.step_cursor != 0 &&
      params.page_tokens_for_step != 0u) {
    const unsigned int position = params.step_cursor[0];
    unsigned int active_pages =
        (position + params.page_tokens_for_step) / params.page_tokens_for_step;
    if (active_pages == 0u) {
      active_pages = 1u;
    }
    pages = active_pages < params.pages ? active_pages : params.pages;
    current_page = position / params.page_tokens_for_step;
    if (current_page >= pages) {
      current_page = pages - 1u;
    }
  }
  const unsigned int sink_pages = min_u32(params.sink_pages, pages);
  const unsigned int raw_local_pages = min_u32(params.local_pages, pages);
  unsigned int local_start =
      current_page + 1u > raw_local_pages
          ? current_page + 1u - raw_local_pages
          : 0u;
  if (local_start < sink_pages) {
    local_start = sink_pages;
  }
  const unsigned int local_pages =
      current_page >= local_start ? current_page - local_start + 1u : 0u;
  const unsigned int local_limit = sink_pages + local_pages;

  unsigned int target = 0u;
  const unsigned int descriptor_geometry =
      params.descriptor_geometry != 0u && params.query_derived_pages != 0u &&
      params.query_dims >= 2u && params.queries != 0;
  if (slot < sink_pages) {
    target = slot;
  } else if (slot < local_limit && local_pages != 0u) {
    target = local_start + (slot - sink_pages);
  } else {
    const unsigned int far_start = sink_pages;
    const unsigned int far_end = local_start;
    const unsigned int far_pages =
        far_end > far_start ? far_end - far_start : 0u;
    if (far_pages == 0u) {
      target = slot % pages;
    } else {
      const unsigned int far_slot = slot - local_limit;
      const unsigned int far_candidates =
          params.candidates_per_query > local_limit
              ? params.candidates_per_query - local_limit
              : 1u;
      unsigned int center = 0u;
      if (params.query_derived_pages != 0u) {
        const unsigned int global_center =
            query_descriptor_center_page(query, pages);
        if (global_center < far_start) {
          center = 0u;
        } else if (global_center >= far_end) {
          center = far_pages - 1u;
        } else {
          center = global_center - far_start;
        }
      } else {
        center = query_center_page(query, far_pages, far_candidates);
      }
      const unsigned int half = far_candidates / 2u;
      target = far_start + ((center + far_pages + far_slot - half) % far_pages);
    }
  }
  const unsigned long long out_index =
      static_cast<unsigned long long>(query) * params.candidates_per_query + slot;
  if (descriptor_geometry &&
      (slot < local_limit || local_start <= sink_pages)) {
    params.candidate_pages[out_index] = target < pages ? target : pages - 1u;
    return;
  }
  float x = 0.0f;
  float y = 0.0f;
  float z = 1.0f;
  if (descriptor_geometry) {
    const unsigned long long query_base =
        static_cast<unsigned long long>(query) * params.query_dims;
    const unsigned int far_slot =
        slot >= local_limit ? slot - local_limit : 0u;
    const unsigned int descriptor_pairs =
        params.query_dims >= 2u ? params.query_dims / 2u : 1u;
    const unsigned int descriptor_pair =
        descriptor_pairs == 0u ? 0u : far_slot % descriptor_pairs;
    const unsigned int descriptor_dim = descriptor_pair * 2u;
    const float qx = fminf(
        1.0f, fmaxf(-1.0f, params.queries[query_base + descriptor_dim]));
    const float qy = fminf(
        1.0f, fmaxf(-1.0f,
                    params.queries[query_base + descriptor_dim + 1ull]));
    const float angle = static_cast<float>(far_slot) * 2.39996323f;
    const float radius = far_slot == 0u ? 0.0f : 0.11f + 0.035f * far_slot;
    x = qx * params.descriptor_scale + cosf(angle) * radius;
    y = qy * params.descriptor_scale + sinf(angle) * radius;
    const unsigned int layer =
        params.layer_index < params.layer_count ? params.layer_index : 0u;
    const unsigned int plane =
        (layer * params.query_count + query) * descriptor_pairs +
        descriptor_pair;
    z = static_cast<float>(plane) * params.descriptor_plane_stride + 1.0f;
  } else {
    const unsigned int x_index = target % params.grid_width;
    const unsigned int y_index = target / params.grid_width;
    x = static_cast<float>(x_index) * params.cell_size;
    y = static_cast<float>(y_index) * params.cell_size;
  }

  float3 origin;
  origin.x = x;
  origin.y = y;
  origin.z = z;
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
  params.candidate_pages[out_index] = hit % pages;
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

const char *nvrtc_host_arch_define() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return "-D__aarch64__";
#elif defined(__x86_64__) || defined(_M_X64)
  return "-D__x86_64";
#else
  return "-DNERVA_UNKNOWN_HOST_ARCH";
#endif
}

bool compile_optix_candidate_input(
    const NervaCudaExperimentalRtCandidateBenchResult *out,
    std::string *input, NervaCudaExperimentalRtCandidateBenchResult *result) {
  nvrtcProgram program = nullptr;
  nvrtcResult nvrtc = nvrtcCreateProgram(&program, kOptixCandidateDeviceSource,
                                         "nerva_experimental_rt_optix.cu", 0,
                                         nullptr, nullptr);
  if (nvrtc != NVRTC_SUCCESS) {
    set_nvrtc_reason(result, "nvrtcCreateProgram", nvrtc);
    return false;
  }

  char arch[64];
  snprintf(arch, sizeof(arch), "compute_75");
  std::string optix_include = std::string("-I") + NERVA_OPTIX_INCLUDE_DIR;
  std::string cuda_include = std::string("-I") + NERVA_CUDA_INCLUDE_DIR;
  const char *options[] = {
      "-std=c++17",
      "-arch",
      arch,
      optix_include.c_str(),
      cuda_include.c_str(),
      "--optix-ir",
      "-lineinfo",
      "-use_fast_math",
      "-default-device",
      "-rdc",
      "true",
      nvrtc_host_arch_define(),
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

  size_t input_size = 0;
  nvrtc = nvrtcGetOptiXIRSize(program, &input_size);
  if (nvrtc != NVRTC_SUCCESS || input_size == 0) {
    set_nvrtc_reason(result, "nvrtcGetOptiXIRSize", nvrtc);
    nvrtcDestroyProgram(&program);
    return false;
  }
  input->assign(input_size, '\0');
  nvrtc = nvrtcGetOptiXIR(program, input->data());
  nvrtcDestroyProgram(&program);
  if (nvrtc != NVRTC_SUCCESS) {
    set_nvrtc_reason(result, "nvrtcGetOptiXIR", nvrtc);
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
    NervaCudaExperimentalRtCandidateBenchResult *out,
    const float *queries = nullptr, uint32_t query_dims = 0,
    uint32_t query_derived_pages = 0,
    const uint32_t *step_cursor = nullptr,
    const float *page_descriptors = nullptr,
    uint32_t page_descriptor_dims = 0, uint32_t layer_count = 1) {
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

  const bool descriptor_geometry =
      page_descriptors != nullptr && page_descriptor_dims >= 2u &&
      queries != nullptr && query_dims >= 2u && layer_count != 0u;
  const uint32_t grid_width = ceil_sqrt_u32(request->pages);
  constexpr float cell_size = 2.0f;
  constexpr float grid_half_size = 0.45f;
  constexpr float descriptor_scale = 16.0f;
  constexpr float descriptor_half_size = 0.42f;
  constexpr float descriptor_plane_stride = 4.0f;
  const uint32_t descriptor_pairs =
      descriptor_geometry ? page_descriptor_dims / 2u : 1u;
  const uint64_t descriptor_entries =
      static_cast<uint64_t>(layer_count) * request->query_count *
      request->pages;
  const uint64_t descriptor_primitives =
      static_cast<uint64_t>(layer_count) * request->query_count *
      descriptor_pairs * request->pages;
  const uint64_t primitive_count =
      descriptor_geometry ? descriptor_primitives : request->pages;
  if (primitive_count == 0 || primitive_count > UINT32_MAX) {
    set_optix_fallback_reason(out, "geometry build",
                              "descriptor primitive count is invalid");
    return false;
  }
  std::vector<float> host_page_descriptors;
  if (descriptor_geometry) {
    const uint64_t descriptor_floats =
        descriptor_entries * page_descriptor_dims;
    if (descriptor_floats == 0 ||
        descriptor_floats > (UINT64_MAX / sizeof(float))) {
      set_optix_fallback_reason(out, "descriptor copy",
                                "descriptor buffer size is invalid");
      return false;
    }
    host_page_descriptors.resize(static_cast<size_t>(descriptor_floats));
    err = cudaMemcpyAsync(host_page_descriptors.data(), page_descriptors,
                          descriptor_floats * sizeof(float),
                          cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) {
      err = cudaStreamSynchronize(stream);
    }
    if (err != cudaSuccess) {
      set_optix_fallback_reason(out, "descriptor copy",
                                cudaGetErrorString(err));
      return false;
    }
  }
  std::vector<RtVertex> vertices(static_cast<size_t>(primitive_count) * 3u);
  for (uint64_t primitive = 0; primitive < primitive_count; ++primitive) {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float half_size = grid_half_size;
    if (descriptor_geometry) {
      const uint32_t page =
          static_cast<uint32_t>(primitive % request->pages);
      const uint32_t pair =
          static_cast<uint32_t>((primitive / request->pages) %
                                descriptor_pairs);
      const uint32_t head =
          static_cast<uint32_t>((primitive /
                                 (static_cast<uint64_t>(request->pages) *
                                  descriptor_pairs)) %
                                request->query_count);
      const uint32_t layer =
          static_cast<uint32_t>(primitive /
                                (static_cast<uint64_t>(request->pages) *
                                 descriptor_pairs *
                                 request->query_count));
      const uint64_t descriptor_index =
          (((static_cast<uint64_t>(layer) * request->query_count + head) *
            request->pages) +
           page) *
              page_descriptor_dims +
          static_cast<uint64_t>(pair) * 2ull;
      x = host_page_descriptors[descriptor_index] * descriptor_scale;
      y = host_page_descriptors[descriptor_index + 1ull] * descriptor_scale;
      z = static_cast<float>(
              (layer * request->query_count + head) * descriptor_pairs + pair) *
          descriptor_plane_stride;
      half_size = descriptor_half_size;
    } else {
      const uint32_t page = static_cast<uint32_t>(primitive);
      x = static_cast<float>(page % grid_width) * cell_size;
      y = static_cast<float>(page / grid_width) * cell_size;
    }
    vertices[static_cast<size_t>(primitive) * 3u + 0u] =
        RtVertex{x - half_size, y - half_size, z};
    vertices[static_cast<size_t>(primitive) * 3u + 1u] =
        RtVertex{x + half_size, y - half_size, z};
    vertices[static_cast<size_t>(primitive) * 3u + 2u] =
        RtVertex{x, y + half_size, z};
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
  state->traversable = gas_handle;
  state->grid_width = grid_width;
  state->cell_size = cell_size;
  state->queries = queries;
  state->step_cursor = step_cursor;
  state->query_dims = query_dims;
  state->query_derived_pages =
      queries != nullptr && query_dims >= 2u && query_derived_pages != 0u
          ? 1u
          : 0u;
  state->descriptor_geometry = descriptor_geometry ? 1u : 0u;
  state->layer_count = layer_count == 0u ? 1u : layer_count;
  state->dynamic_step = step_cursor != nullptr ? 1u : 0u;
  state->descriptor_scale = descriptor_scale;
  state->descriptor_plane_stride = descriptor_plane_stride;

  std::string optix_input;
  if (!compile_optix_candidate_input(out, &optix_input, out)) {
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
                            optix_input.data(), optix_input.size(), log, &log_size,
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
  params.handle = state->traversable;
  params.candidate_pages = candidate_pages;
  params.queries = state->queries;
  params.step_cursor = state->step_cursor;
  params.pages = request->pages;
  params.query_count = request->query_count;
  params.candidates_per_query = request->candidates_per_query;
  params.layer_index = 0;
  params.layer_count = state->layer_count;
  params.query_dims = state->query_dims;
  params.query_derived_pages = state->query_derived_pages;
  params.descriptor_geometry = state->descriptor_geometry;
  params.page_tokens_for_step = request->page_tokens;
  params.dynamic_step = state->dynamic_step;
  params.grid_width = state->grid_width;
  params.current_page = request->pages - 1u;
  params.local_pages = 0;
  params.sink_pages = 0;
  params.cell_size = state->cell_size;
  params.descriptor_scale = state->descriptor_scale;
  params.descriptor_plane_stride = state->descriptor_plane_stride;
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

__device__ uint32_t query_center_page(uint32_t query, uint32_t pages,
                                      uint32_t candidates_per_query) {
  if (pages == 0u) {
    return 0u;
  }
  const uint32_t seed = hash32(query * 977u + 31u);
  if (candidates_per_query > 0u && candidates_per_query < pages) {
    const uint32_t half = candidates_per_query / 2u;
    const uint32_t span = pages - candidates_per_query + 1u;
    return half + (seed % span);
  }
  return seed % pages;
}

__device__ float page_position(uint32_t page, uint32_t pages) {
  return pages <= 1u ? 0.0f
                     : static_cast<float>(page) /
                           static_cast<float>(pages - 1u);
}

__device__ float synthetic_score_scale(uint32_t pages) {
  constexpr float kSpreadPages = 128.0f;
  const float extent = pages <= 1u ? 1.0f : static_cast<float>(pages - 1u);
  const float scaled = extent / kSpreadPages;
  return 0.5f * scaled * scaled;
}

__device__ float token_position(uint32_t token_offset, uint32_t page_tokens) {
  if (page_tokens <= 1u) {
    return 0.0f;
  }
  return 2.0f * static_cast<float>(token_offset) /
             static_cast<float>(page_tokens - 1u) -
         1.0f;
}

__device__ uint32_t query_target_token_offset(uint32_t query,
                                              uint32_t page_tokens) {
  if (page_tokens <= 2u) {
    return 0u;
  }
  const uint32_t interior_tokens = page_tokens - 2u;
  return 1u + (hash32(query * 747796405u + 2891336453u) % interior_tokens);
}

__device__ float synthetic_token_score_scale(uint32_t page_tokens) {
  (void)page_tokens;
  return 1.0f;
}

__global__ void init_page_descriptors_kernel(float *descriptors, uint32_t pages,
                                             uint32_t dims) {
  const uint64_t total = static_cast<uint64_t>(pages) * dims;
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  while (index < total) {
    const uint32_t page = static_cast<uint32_t>(index / dims);
    const uint32_t dim = static_cast<uint32_t>(index - static_cast<uint64_t>(page) * dims);
    const float position = page_position(page, pages);
    if (dim == 0u) {
      descriptors[index] = position;
    } else if (dim == 1u) {
      descriptors[index] = position * position;
    } else {
      descriptors[index] = 0.0f;
    }
    index += stride;
  }
}

__global__ void init_query_descriptors_kernel(float *queries, uint32_t query_count,
                                              uint32_t pages, uint32_t dims,
                                              uint32_t candidates_per_query,
                                              uint32_t page_tokens) {
  const uint64_t total = static_cast<uint64_t>(query_count) * dims;
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  while (index < total) {
    const uint32_t query = static_cast<uint32_t>(index / dims);
    const uint32_t dim = static_cast<uint32_t>(index - static_cast<uint64_t>(query) * dims);
    const uint32_t center =
        query_center_page(query, pages, candidates_per_query);
    const float center_position = page_position(center, pages);
    const float scale = synthetic_score_scale(pages) *
                        sqrtf(static_cast<float>(dims));
    const uint32_t target_token =
        query_target_token_offset(query, page_tokens);
    const float target_token_position =
        token_position(target_token, page_tokens);
    const float token_scale = synthetic_token_score_scale(page_tokens) *
                              sqrtf(static_cast<float>(dims));
    if (dim == 0u) {
      queries[index] = 2.0f * center_position * scale;
    } else if (dim == 1u) {
      queries[index] = -scale;
    } else if (dim == 2u) {
      queries[index] = 2.0f * target_token_position * token_scale;
    } else if (dim == 3u) {
      queries[index] = -token_scale;
    } else {
      queries[index] = 0.0f;
    }
    index += stride;
  }
}

__global__ void init_kv_cache_kernel(float *keys, float *values, uint32_t pages,
                                     uint32_t page_tokens, uint32_t dims) {
  const uint64_t total =
      static_cast<uint64_t>(pages) * page_tokens * dims;
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  while (index < total) {
    const uint32_t dim = static_cast<uint32_t>(index % dims);
    const uint64_t token = index / dims;
    const uint32_t page = static_cast<uint32_t>(token / page_tokens);
    const uint32_t token_offset = static_cast<uint32_t>(token % page_tokens);
    const float position = page_position(page, pages);
    const float position_in_page = token_position(token_offset, page_tokens);
    if (dim == 0u) {
      keys[index] = position;
    } else if (dim == 1u) {
      keys[index] = position * position;
    } else if (dim == 2u) {
      keys[index] = position_in_page;
    } else if (dim == 3u) {
      keys[index] = position_in_page * position_in_page;
    } else {
      keys[index] = 0.0f;
    }
    values[index] =
        deterministic_f32(static_cast<uint32_t>(token + 0x51u), dim + 193u);
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

__device__ float kv_dot(const float *keys, const float *queries, uint32_t page,
                        uint32_t token_offset, uint32_t query, uint32_t dims,
                        uint32_t page_tokens) {
  float sum = 0.0f;
  const uint64_t key_base =
      (static_cast<uint64_t>(page) * page_tokens + token_offset) * dims;
  const uint64_t query_base = static_cast<uint64_t>(query) * dims;
  for (uint32_t dim = 0; dim < dims; ++dim) {
    sum += keys[key_base + dim] * queries[query_base + dim];
  }
  return sum * rsqrtf(static_cast<float>(dims));
}

__device__ float projected_kv_dot(const float *keys, const float *queries,
                                  uint32_t page, uint32_t token_offset,
                                  uint32_t query, uint32_t dims,
                                  uint32_t page_tokens,
                                  uint32_t projection_dims) {
  float sum = 0.0f;
  const uint32_t capped_projection_dims =
      dims < projection_dims ? dims : projection_dims;
  const uint64_t key_base =
      (static_cast<uint64_t>(page) * page_tokens + token_offset) * dims;
  const uint64_t query_base = static_cast<uint64_t>(query) * dims;
  for (uint32_t dim = 0; dim < capped_projection_dims; ++dim) {
    sum += keys[key_base + dim] * queries[query_base + dim];
  }
  return sum * rsqrtf(static_cast<float>(dims));
}

__device__ uint32_t norm_stress_page(uint32_t query, uint32_t pages,
                                     uint32_t page_tokens,
                                     uint64_t local_window_tokens) {
  if (pages == 0u || page_tokens == 0u) {
    return 0u;
  }
  const uint64_t total_tokens = static_cast<uint64_t>(pages) * page_tokens;
  const uint64_t local_tokens =
      total_tokens < local_window_tokens ? total_tokens : local_window_tokens;
  const uint64_t local_start = total_tokens - local_tokens;
  const uint32_t far_pages =
      static_cast<uint32_t>(local_start / page_tokens);
  if (far_pages == 0u) {
    return 0u;
  }
  const uint32_t center = query_center_page(query, pages, 1u) % far_pages;
  const uint32_t offset = (far_pages / 3u) + 1u;
  return (center + offset) % far_pages;
}

__device__ float norm_stress_score(const float *keys, const float *queries,
                                   uint32_t page, uint32_t token_offset,
                                   uint32_t query, uint32_t dims,
                                   uint32_t page_tokens,
                                   uint32_t stress_page) {
  const float base =
      kv_dot(keys, queries, page, token_offset, query, dims, page_tokens);
  if (page != stress_page) {
    return base;
  }
  const float token_tiebreak =
      static_cast<float>(page_tokens - 1u - token_offset) * 0.0001f;
  return base + 1000000.0f + token_tiebreak;
}

__device__ void reduce_attention_block(float *shared_scores,
                                       float *shared_output, uint32_t dims,
                                       float local_sum,
                                       const float *local_output,
                                       float *out, uint32_t query,
                                       float max_score, float *meta) {
  shared_scores[threadIdx.x] = local_sum;
  for (uint32_t dim = 0; dim < dims; ++dim) {
    shared_output[static_cast<uint32_t>(threadIdx.x) * kMaxAttentionDims + dim] =
        local_output[dim];
  }
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_scores[threadIdx.x] += shared_scores[threadIdx.x + stride];
      for (uint32_t dim = 0; dim < dims; ++dim) {
        shared_output[static_cast<uint32_t>(threadIdx.x) * kMaxAttentionDims + dim] +=
            shared_output[static_cast<uint32_t>(threadIdx.x + stride) *
                              kMaxAttentionDims +
                          dim];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float denom = shared_scores[0] > 0.0f ? shared_scores[0] : 1.0f;
    const uint64_t out_base = static_cast<uint64_t>(query) * dims;
    for (uint32_t dim = 0; dim < dims; ++dim) {
      out[out_base + dim] = shared_output[dim] / denom;
    }
    if (meta != nullptr) {
      meta[static_cast<uint64_t>(query) * 2ull] = max_score;
      meta[static_cast<uint64_t>(query) * 2ull + 1ull] = denom;
    }
  }
}

__global__ void local_attention_kernel(const float *keys, const float *values,
                                       const float *queries, uint32_t pages,
                                       uint32_t page_tokens, uint32_t dims,
                                       uint32_t query_count,
                                       uint64_t local_window_tokens,
                                       float *out, float *meta) {
  const uint32_t query = blockIdx.x;
  if (query >= query_count || dims > kMaxAttentionDims) {
    return;
  }
  const uint64_t total_tokens = static_cast<uint64_t>(pages) * page_tokens;
  const uint64_t tokens =
      total_tokens < local_window_tokens ? total_tokens : local_window_tokens;
  const uint64_t token_start = total_tokens - tokens;
  float local_max = -INFINITY;
  for (uint64_t index = threadIdx.x; index < tokens; index += blockDim.x) {
    const uint64_t token = token_start + index;
    const uint32_t page = static_cast<uint32_t>(token / page_tokens);
    const uint32_t token_offset = static_cast<uint32_t>(token % page_tokens);
    local_max = fmaxf(local_max,
                      kv_dot(keys, queries, page, token_offset, query, dims,
                             page_tokens));
  }
  __shared__ float shared_scores[kThreads];
  __shared__ float shared_output[kThreads * kMaxAttentionDims];
  shared_scores[threadIdx.x] = local_max;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_scores[threadIdx.x] =
          fmaxf(shared_scores[threadIdx.x], shared_scores[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_score = shared_scores[0];
  float local_sum = 0.0f;
  float local_output[kMaxAttentionDims];
  for (uint32_t dim = 0; dim < kMaxAttentionDims; ++dim) {
    local_output[dim] = 0.0f;
  }
  for (uint64_t index = threadIdx.x; index < tokens; index += blockDim.x) {
    const uint64_t token = token_start + index;
    const uint32_t page = static_cast<uint32_t>(token / page_tokens);
    const uint32_t token_offset = static_cast<uint32_t>(token % page_tokens);
    const float score =
        kv_dot(keys, queries, page, token_offset, query, dims, page_tokens);
    const float weight = expf(score - max_score);
    local_sum += weight;
    const uint64_t value_base =
        (static_cast<uint64_t>(page) * page_tokens + token_offset) * dims;
    for (uint32_t dim = 0; dim < dims; ++dim) {
      local_output[dim] += weight * values[value_base + dim];
    }
  }
  reduce_attention_block(shared_scores, shared_output, dims, local_sum,
                         local_output, out, query, max_score, meta);
}

__global__ void attention_mass_recall_kernel(
    const float *keys, const float *queries, const uint32_t *candidate_pages,
    uint32_t pages, uint32_t page_tokens, uint32_t dims, uint32_t query_count,
    uint32_t candidates_per_query, uint64_t local_window_tokens,
    uint64_t *recall_ppm) {
  const uint32_t query = blockIdx.x;
  if (query >= query_count || dims > kMaxAttentionDims) {
    return;
  }
  const uint64_t total_tokens = static_cast<uint64_t>(pages) * page_tokens;
  const uint64_t local_tokens =
      total_tokens < local_window_tokens ? total_tokens : local_window_tokens;
  const uint64_t local_start = total_tokens - local_tokens;

  float local_max = -INFINITY;
  for (uint64_t token = threadIdx.x; token < total_tokens;
       token += blockDim.x) {
    const uint32_t page = static_cast<uint32_t>(token / page_tokens);
    const uint32_t token_offset = static_cast<uint32_t>(token % page_tokens);
    local_max = fmaxf(local_max,
                      kv_dot(keys, queries, page, token_offset, query, dims,
                             page_tokens));
  }
  __shared__ float shared_max[kThreads];
  shared_max[threadIdx.x] = local_max;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_max[threadIdx.x] =
          fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_score = shared_max[0];

  double total_sum = 0.0;
  double selected_sum = 0.0;
  for (uint64_t token = threadIdx.x; token < total_tokens;
       token += blockDim.x) {
    const uint32_t page = static_cast<uint32_t>(token / page_tokens);
    const uint32_t token_offset = static_cast<uint32_t>(token % page_tokens);
    const float score =
        kv_dot(keys, queries, page, token_offset, query, dims, page_tokens);
    const double weight = static_cast<double>(expf(score - max_score));
    total_sum += weight;
    if (token >= local_start) {
      selected_sum += weight;
    }
  }

  const uint64_t candidate_tokens =
      static_cast<uint64_t>(candidates_per_query) * page_tokens;
  for (uint64_t index = threadIdx.x; index < candidate_tokens;
       index += blockDim.x) {
    const uint32_t candidate = static_cast<uint32_t>(index / page_tokens);
    const uint32_t token_offset = static_cast<uint32_t>(index % page_tokens);
    const uint64_t candidate_base =
        static_cast<uint64_t>(query) * candidates_per_query;
    const uint32_t page = candidate_pages[candidate_base + candidate] % pages;
    bool duplicate = false;
    for (uint32_t previous = 0; previous < candidate; ++previous) {
      if ((candidate_pages[candidate_base + previous] % pages) == page) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      continue;
    }
    const uint64_t token =
        static_cast<uint64_t>(page) * page_tokens + token_offset;
    if (token >= local_start) {
      continue;
    }
    const float score =
        kv_dot(keys, queries, page, token_offset, query, dims, page_tokens);
    selected_sum += static_cast<double>(expf(score - max_score));
  }

  __shared__ double shared_total[kThreads];
  __shared__ double shared_selected[kThreads];
  shared_total[threadIdx.x] = total_sum;
  shared_selected[threadIdx.x] = selected_sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_total[threadIdx.x] += shared_total[threadIdx.x + stride];
      shared_selected[threadIdx.x] += shared_selected[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    uint64_t ppm = 0;
    if (shared_total[0] > 0.0) {
      double recall = shared_selected[0] / shared_total[0];
      if (recall < 0.0) {
        recall = 0.0;
      } else if (recall > 1.0) {
        recall = 1.0;
      }
      ppm = static_cast<uint64_t>(recall * 1000000.0 + 0.5);
    }
    recall_ppm[query] = ppm;
  }
}

__global__ void far_oracle_topk_diagnostics_kernel(
    const float *keys, const float *queries, const uint32_t *candidate_pages,
    uint32_t pages, uint32_t page_tokens, uint32_t dims,
    uint32_t query_count, uint32_t candidates_per_query,
    uint64_t local_window_tokens, uint32_t topk_tokens,
    uint64_t *token_recall_ppm, uint32_t *scatter_pages) {
  const uint32_t query = blockIdx.x;
  if (query >= query_count || dims > kMaxAttentionDims) {
    return;
  }
  const uint64_t total_tokens = static_cast<uint64_t>(pages) * page_tokens;
  const uint64_t local_tokens =
      total_tokens < local_window_tokens ? total_tokens : local_window_tokens;
  const uint64_t far_tokens = total_tokens - local_tokens;
  if (far_tokens == 0 || topk_tokens == 0) {
    if (threadIdx.x == 0) {
      token_recall_ppm[query] = 0;
      scatter_pages[query] = 0;
    }
    return;
  }

  __shared__ float scores[kThreads];
  __shared__ uint64_t tokens_shared[kThreads];
  __shared__ uint64_t selected_tokens[kMaxOracleTopK];
  __shared__ uint32_t selected_pages[kMaxOracleTopK];
  __shared__ uint32_t selected_count_shared;
  if (threadIdx.x == 0) {
    selected_count_shared = 0;
  }
  __syncthreads();

  const uint32_t capped_topk =
      topk_tokens < kMaxOracleTopK ? topk_tokens : kMaxOracleTopK;
  for (uint32_t rank = 0; rank < capped_topk; ++rank) {
    float best_score = -INFINITY;
    uint64_t best_token = UINT64_MAX;
    for (uint64_t token = threadIdx.x; token < far_tokens;
         token += blockDim.x) {
      bool selected = false;
      for (uint32_t previous = 0; previous < rank; ++previous) {
        selected = selected || selected_tokens[previous] == token;
      }
      if (selected) {
        continue;
      }
      const uint32_t page = static_cast<uint32_t>(token / page_tokens);
      const uint32_t token_offset =
          static_cast<uint32_t>(token % page_tokens);
      const float score =
          kv_dot(keys, queries, page, token_offset, query, dims, page_tokens);
      if (score > best_score ||
          (score == best_score && token < best_token)) {
        best_score = score;
        best_token = token;
      }
    }
    scores[threadIdx.x] = best_score;
    tokens_shared[threadIdx.x] = best_token;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        const float other_score = scores[threadIdx.x + stride];
        const uint64_t other_token = tokens_shared[threadIdx.x + stride];
        if (other_score > scores[threadIdx.x] ||
            (other_score == scores[threadIdx.x] &&
             other_token < tokens_shared[threadIdx.x])) {
          scores[threadIdx.x] = other_score;
          tokens_shared[threadIdx.x] = other_token;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      selected_tokens[rank] = tokens_shared[0];
      selected_pages[rank] =
          static_cast<uint32_t>(tokens_shared[0] / page_tokens);
      selected_count_shared = rank + 1;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    uint32_t distinct_pages = 0;
    uint32_t captured_tokens = 0;
    const uint64_t candidate_base =
        static_cast<uint64_t>(query) * candidates_per_query;
    for (uint32_t rank = 0; rank < selected_count_shared; ++rank) {
      bool duplicate = false;
      for (uint32_t previous = 0; previous < rank; ++previous) {
        duplicate = duplicate || selected_pages[previous] == selected_pages[rank];
      }
      if (!duplicate) {
        ++distinct_pages;
      }
      bool captured = false;
      for (uint32_t candidate = 0; candidate < candidates_per_query; ++candidate) {
        captured = captured ||
                   (candidate_pages[candidate_base + candidate] % pages) ==
                       selected_pages[rank];
      }
      if (captured) {
        ++captured_tokens;
      }
    }
    token_recall_ppm[query] =
        selected_count_shared == 0
            ? 0
            : (static_cast<uint64_t>(captured_tokens) * 1000000ull) /
                  selected_count_shared;
    scatter_pages[query] = distinct_pages;
  }
}

__global__ void fine_token_projected_topk_diagnostics_kernel(
    const float *keys, const float *queries, uint32_t pages,
    uint32_t page_tokens, uint32_t dims, uint32_t query_count,
    uint64_t local_window_tokens, uint32_t topk_tokens,
    uint32_t candidate_tokens, uint32_t projection_dims,
    uint64_t *token_recall_ppm) {
  const uint32_t query = blockIdx.x;
  if (query >= query_count || dims > kMaxAttentionDims) {
    return;
  }
  const uint64_t total_tokens = static_cast<uint64_t>(pages) * page_tokens;
  const uint64_t local_tokens =
      total_tokens < local_window_tokens ? total_tokens : local_window_tokens;
  const uint64_t far_tokens = total_tokens - local_tokens;
  if (far_tokens == 0 || topk_tokens == 0 || candidate_tokens == 0) {
    if (threadIdx.x == 0) {
      token_recall_ppm[query] = 0;
    }
    return;
  }

  __shared__ float scores[kThreads];
  __shared__ uint64_t tokens_shared[kThreads];
  __shared__ uint64_t oracle_tokens[kMaxOracleTopK];
  __shared__ uint64_t projected_tokens[kMaxOracleTopK];
  __shared__ uint32_t oracle_count_shared;
  __shared__ uint32_t projected_count_shared;
  if (threadIdx.x == 0) {
    oracle_count_shared = 0;
    projected_count_shared = 0;
  }
  __syncthreads();

  const uint32_t capped_topk =
      topk_tokens < kMaxOracleTopK ? topk_tokens : kMaxOracleTopK;
  for (uint32_t rank = 0; rank < capped_topk; ++rank) {
    float best_score = -INFINITY;
    uint64_t best_token = UINT64_MAX;
    for (uint64_t token = threadIdx.x; token < far_tokens;
         token += blockDim.x) {
      bool selected = false;
      for (uint32_t previous = 0; previous < rank; ++previous) {
        selected = selected || oracle_tokens[previous] == token;
      }
      if (selected) {
        continue;
      }
      const uint32_t page = static_cast<uint32_t>(token / page_tokens);
      const uint32_t token_offset =
          static_cast<uint32_t>(token % page_tokens);
      const float score =
          kv_dot(keys, queries, page, token_offset, query, dims, page_tokens);
      if (score > best_score ||
          (score == best_score && token < best_token)) {
        best_score = score;
        best_token = token;
      }
    }
    scores[threadIdx.x] = best_score;
    tokens_shared[threadIdx.x] = best_token;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        const float other_score = scores[threadIdx.x + stride];
        const uint64_t other_token = tokens_shared[threadIdx.x + stride];
        if (other_score > scores[threadIdx.x] ||
            (other_score == scores[threadIdx.x] &&
             other_token < tokens_shared[threadIdx.x])) {
          scores[threadIdx.x] = other_score;
          tokens_shared[threadIdx.x] = other_token;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      oracle_tokens[rank] = tokens_shared[0];
      oracle_count_shared = rank + 1;
    }
    __syncthreads();
  }

  const uint32_t capped_candidates =
      candidate_tokens < kMaxOracleTopK ? candidate_tokens : kMaxOracleTopK;
  for (uint32_t rank = 0; rank < capped_candidates; ++rank) {
    float best_score = -INFINITY;
    uint64_t best_token = UINT64_MAX;
    for (uint64_t token = threadIdx.x; token < far_tokens;
         token += blockDim.x) {
      bool selected = false;
      for (uint32_t previous = 0; previous < rank; ++previous) {
        selected = selected || projected_tokens[previous] == token;
      }
      if (selected) {
        continue;
      }
      const uint32_t page = static_cast<uint32_t>(token / page_tokens);
      const uint32_t token_offset =
          static_cast<uint32_t>(token % page_tokens);
      const float score =
          projected_kv_dot(keys, queries, page, token_offset, query, dims,
                           page_tokens, projection_dims);
      if (score > best_score ||
          (score == best_score && token < best_token)) {
        best_score = score;
        best_token = token;
      }
    }
    scores[threadIdx.x] = best_score;
    tokens_shared[threadIdx.x] = best_token;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        const float other_score = scores[threadIdx.x + stride];
        const uint64_t other_token = tokens_shared[threadIdx.x + stride];
        if (other_score > scores[threadIdx.x] ||
            (other_score == scores[threadIdx.x] &&
             other_token < tokens_shared[threadIdx.x])) {
          scores[threadIdx.x] = other_score;
          tokens_shared[threadIdx.x] = other_token;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      projected_tokens[rank] = tokens_shared[0];
      projected_count_shared = rank + 1;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    uint32_t captured_tokens = 0;
    for (uint32_t rank = 0; rank < oracle_count_shared; ++rank) {
      bool captured = false;
      for (uint32_t candidate = 0; candidate < projected_count_shared;
           ++candidate) {
        captured = captured || projected_tokens[candidate] == oracle_tokens[rank];
      }
      if (captured) {
        ++captured_tokens;
      }
    }
    token_recall_ppm[query] =
        oracle_count_shared == 0
            ? 0
            : (static_cast<uint64_t>(captured_tokens) * 1000000ull) /
                  oracle_count_shared;
  }
}

__global__ void norm_stress_topk_diagnostics_kernel(
    const float *keys, const float *queries, const uint32_t *candidate_pages,
    uint32_t pages, uint32_t page_tokens, uint32_t dims,
    uint32_t query_count, uint32_t candidates_per_query,
    uint64_t local_window_tokens, uint32_t topk_tokens,
    uint64_t *no_augmentation_recall_ppm,
    uint64_t *synthetic_norm_augmented_recall_ppm) {
  const uint32_t query = blockIdx.x;
  if (query >= query_count || dims > kMaxAttentionDims) {
    return;
  }
  const uint64_t total_tokens = static_cast<uint64_t>(pages) * page_tokens;
  const uint64_t local_tokens =
      total_tokens < local_window_tokens ? total_tokens : local_window_tokens;
  const uint64_t far_tokens = total_tokens - local_tokens;
  if (far_tokens == 0 || topk_tokens == 0) {
    if (threadIdx.x == 0) {
      no_augmentation_recall_ppm[query] = 0;
      synthetic_norm_augmented_recall_ppm[query] = 0;
    }
    return;
  }

  __shared__ float scores[kThreads];
  __shared__ uint64_t tokens_shared[kThreads];
  __shared__ uint64_t selected_tokens[kMaxOracleTopK];
  __shared__ uint32_t selected_pages[kMaxOracleTopK];
  __shared__ uint32_t selected_count_shared;
  if (threadIdx.x == 0) {
    selected_count_shared = 0;
  }
  __syncthreads();

  const uint32_t stress_page =
      norm_stress_page(query, pages, page_tokens, local_window_tokens);
  const uint32_t capped_topk =
      topk_tokens < kMaxOracleTopK ? topk_tokens : kMaxOracleTopK;
  for (uint32_t rank = 0; rank < capped_topk; ++rank) {
    float best_score = -INFINITY;
    uint64_t best_token = UINT64_MAX;
    for (uint64_t token = threadIdx.x; token < far_tokens;
         token += blockDim.x) {
      bool selected = false;
      for (uint32_t previous = 0; previous < rank; ++previous) {
        selected = selected || selected_tokens[previous] == token;
      }
      if (selected) {
        continue;
      }
      const uint32_t page = static_cast<uint32_t>(token / page_tokens);
      const uint32_t token_offset =
          static_cast<uint32_t>(token % page_tokens);
      const float score = norm_stress_score(keys, queries, page, token_offset,
                                            query, dims, page_tokens,
                                            stress_page);
      if (score > best_score ||
          (score == best_score && token < best_token)) {
        best_score = score;
        best_token = token;
      }
    }
    scores[threadIdx.x] = best_score;
    tokens_shared[threadIdx.x] = best_token;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        const float other_score = scores[threadIdx.x + stride];
        const uint64_t other_token = tokens_shared[threadIdx.x + stride];
        if (other_score > scores[threadIdx.x] ||
            (other_score == scores[threadIdx.x] &&
             other_token < tokens_shared[threadIdx.x])) {
          scores[threadIdx.x] = other_score;
          tokens_shared[threadIdx.x] = other_token;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      selected_tokens[rank] = tokens_shared[0];
      selected_pages[rank] =
          static_cast<uint32_t>(tokens_shared[0] / page_tokens);
      selected_count_shared = rank + 1;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    uint32_t no_augmented_captured = 0;
    uint32_t norm_augmented_captured = 0;
    const uint64_t candidate_base =
        static_cast<uint64_t>(query) * candidates_per_query;
    for (uint32_t rank = 0; rank < selected_count_shared; ++rank) {
      bool no_augmented_page = false;
      bool norm_augmented_page = selected_pages[rank] == stress_page;
      for (uint32_t candidate = 0; candidate < candidates_per_query; ++candidate) {
        const uint32_t page =
            candidate_pages[candidate_base + candidate] % pages;
        no_augmented_page = no_augmented_page || page == selected_pages[rank];
        if (candidate + 1u < candidates_per_query) {
          norm_augmented_page =
              norm_augmented_page || page == selected_pages[rank];
        }
      }
      if (no_augmented_page) {
        ++no_augmented_captured;
      }
      if (norm_augmented_page) {
        ++norm_augmented_captured;
      }
    }
    no_augmentation_recall_ppm[query] =
        selected_count_shared == 0
            ? 0
            : (static_cast<uint64_t>(no_augmented_captured) * 1000000ull) /
                  selected_count_shared;
    synthetic_norm_augmented_recall_ppm[query] =
        selected_count_shared == 0
            ? 0
            : (static_cast<uint64_t>(norm_augmented_captured) * 1000000ull) /
                  selected_count_shared;
  }
}

__global__ void far_sparse_attention_kernel(
    const float *keys, const float *values, const float *queries,
    const uint32_t *candidate_pages, uint32_t pages, uint32_t page_tokens,
    uint32_t dims, uint32_t query_count, uint32_t candidates_per_query,
    uint64_t local_window_tokens, float *out, float *meta) {
  const uint32_t query = blockIdx.x;
  if (query >= query_count || dims > kMaxAttentionDims) {
    return;
  }
  const uint64_t tokens =
      static_cast<uint64_t>(candidates_per_query) * page_tokens;
  const uint64_t total_tokens = static_cast<uint64_t>(pages) * page_tokens;
  const uint64_t local_tokens =
      total_tokens < local_window_tokens ? total_tokens : local_window_tokens;
  const uint64_t local_start = total_tokens - local_tokens;
  float local_max = -INFINITY;
  for (uint64_t index = threadIdx.x; index < tokens; index += blockDim.x) {
    const uint32_t candidate = static_cast<uint32_t>(index / page_tokens);
    const uint32_t token_offset = static_cast<uint32_t>(index % page_tokens);
    const uint32_t page =
        candidate_pages[static_cast<uint64_t>(query) * candidates_per_query +
                        candidate] %
        pages;
    const uint64_t token =
        static_cast<uint64_t>(page) * page_tokens + token_offset;
    if (token >= local_start) {
      continue;
    }
    local_max = fmaxf(local_max,
                      kv_dot(keys, queries, page, token_offset, query, dims,
                             page_tokens));
  }
  __shared__ float shared_scores[kThreads];
  __shared__ float shared_output[kThreads * kMaxAttentionDims];
  shared_scores[threadIdx.x] = local_max;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_scores[threadIdx.x] =
          fmaxf(shared_scores[threadIdx.x], shared_scores[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_score = shared_scores[0];
  float local_sum = 0.0f;
  float local_output[kMaxAttentionDims];
  for (uint32_t dim = 0; dim < kMaxAttentionDims; ++dim) {
    local_output[dim] = 0.0f;
  }
  for (uint64_t index = threadIdx.x; index < tokens; index += blockDim.x) {
    const uint32_t candidate = static_cast<uint32_t>(index / page_tokens);
    const uint32_t token_offset = static_cast<uint32_t>(index % page_tokens);
    const uint32_t page =
        candidate_pages[static_cast<uint64_t>(query) * candidates_per_query +
                        candidate] %
        pages;
    const uint64_t token =
        static_cast<uint64_t>(page) * page_tokens + token_offset;
    if (token >= local_start) {
      continue;
    }
    const float score =
        kv_dot(keys, queries, page, token_offset, query, dims, page_tokens);
    const float weight = expf(score - max_score);
    local_sum += weight;
    const uint64_t value_base =
        (static_cast<uint64_t>(page) * page_tokens + token_offset) * dims;
    for (uint32_t dim = 0; dim < dims; ++dim) {
      local_output[dim] += weight * values[value_base + dim];
    }
  }
  reduce_attention_block(shared_scores, shared_output, dims, local_sum,
                         local_output, out, query, max_score, meta);
}

__global__ void kv_page_access_kernel(const float *keys, const float *values,
                                      const uint32_t *candidate_pages,
                                      uint32_t pages, uint32_t page_tokens,
                                      uint32_t dims, uint32_t query_count,
                                      uint32_t candidates_per_query,
                                      float *touch_out) {
  const uint32_t query = blockIdx.x;
  if (query >= query_count) {
    return;
  }
  float sum = 0.0f;
  const uint64_t total =
      static_cast<uint64_t>(candidates_per_query) * page_tokens * dims;
  for (uint64_t index = threadIdx.x; index < total; index += blockDim.x) {
    const uint32_t dim = static_cast<uint32_t>(index % dims);
    const uint64_t token_in_pages = index / dims;
    const uint32_t candidate =
        static_cast<uint32_t>(token_in_pages / page_tokens);
    const uint32_t token_offset =
        static_cast<uint32_t>(token_in_pages % page_tokens);
    const uint32_t page =
        candidate_pages[static_cast<uint64_t>(query) * candidates_per_query +
                        candidate] %
        pages;
    const uint64_t base =
        (static_cast<uint64_t>(page) * page_tokens + token_offset) * dims + dim;
    sum += keys[base] + values[base];
  }
  __shared__ float shared[kThreads];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    touch_out[query] = shared[0];
  }
}

__global__ void merge_attention_outputs_kernel(const float *local_out,
                                               const float *local_meta,
                                               const float *far_out,
                                               const float *far_meta,
                                               uint32_t dims,
                                               uint32_t query_count,
                                               float *merged_out) {
  const uint64_t total = static_cast<uint64_t>(query_count) * dims;
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  while (index < total) {
    const uint32_t query = static_cast<uint32_t>(index / dims);
    const float local_max = local_meta[static_cast<uint64_t>(query) * 2ull];
    const float local_denom =
        local_meta[static_cast<uint64_t>(query) * 2ull + 1ull];
    const float far_max = far_meta[static_cast<uint64_t>(query) * 2ull];
    const float far_denom =
        far_meta[static_cast<uint64_t>(query) * 2ull + 1ull];
    const float merged_max = fmaxf(local_max, far_max);
    const float local_weight =
        isfinite(local_max) ? local_denom * expf(local_max - merged_max) : 0.0f;
    const float far_weight =
        isfinite(far_max) ? far_denom * expf(far_max - merged_max) : 0.0f;
    const float denom = local_weight + far_weight;
    merged_out[index] =
        denom > 0.0f
            ? (local_out[index] * local_weight + far_out[index] * far_weight) /
                  denom
            : 0.0f;
    index += stride;
  }
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
  const uint32_t center =
      query_center_page(query, pages, candidates_per_query);
  const uint32_t half = candidates_per_query / 2u;
  for (uint32_t offset = threadIdx.x; offset < candidates_per_query; offset += blockDim.x) {
    const uint32_t wrapped = center + pages + offset - half;
    candidate_pages[static_cast<uint64_t>(query) * candidates_per_query + offset] =
        wrapped % pages;
  }
}

__global__ void page_level_candidate_selector_kernel(
    const float *descriptors, const float *queries, uint32_t *candidate_pages,
    uint32_t pages, uint32_t dims, uint32_t query_count,
    uint32_t candidates_per_query) {
  const uint32_t query = blockIdx.x;
  if (query >= query_count || pages == 0) {
    return;
  }
  const uint64_t base = static_cast<uint64_t>(query) * candidates_per_query;
  for (uint32_t rank = 0; rank < candidates_per_query; ++rank) {
    float best_score = -INFINITY;
    uint32_t best_page = 0;
    for (uint32_t page = threadIdx.x; page < pages; page += blockDim.x) {
      bool selected = false;
      for (uint32_t previous = 0; previous < rank; ++previous) {
        selected = selected || candidate_pages[base + previous] == page;
      }
      if (selected) {
        continue;
      }
      const float score = descriptor_dot(descriptors, queries, page, query, dims);
      if (score > best_score ||
          (score == best_score && page < best_page)) {
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
            (other_score == scores[threadIdx.x] &&
             other_page < pages_shared[threadIdx.x])) {
          scores[threadIdx.x] = other_score;
          pages_shared[threadIdx.x] = other_page;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      candidate_pages[base + rank] = pages_shared[0];
    }
    __syncthreads();
  }
}

__global__ void compare_candidate_pages_kernel(const uint32_t *actual,
                                               const uint32_t *expected,
                                               uint64_t total,
                                               unsigned long long *stats) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  while (index < total) {
    const uint32_t actual_value = actual[index];
    const uint32_t expected_value = expected[index];
    if (actual_value != expected_value) {
      const unsigned long long previous = atomicAdd(&stats[0], 1ull);
      if (previous == 0ull) {
        stats[1] = static_cast<unsigned long long>(index);
        stats[2] = static_cast<unsigned long long>(expected_value);
        stats[3] = static_cast<unsigned long long>(actual_value);
      }
    }
    index += stride;
  }
}

__global__ void hash_candidate_queries_kernel(const uint32_t *candidate_pages,
                                              uint32_t query_count,
                                              uint32_t candidates_per_query,
                                              uint64_t *query_hashes) {
  const uint32_t query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= query_count) {
    return;
  }
  uint64_t hash = 1469598103934665603ull;
  const uint64_t base = static_cast<uint64_t>(query) * candidates_per_query;
  for (uint32_t index = 0; index < candidates_per_query; ++index) {
    hash ^= static_cast<uint64_t>(candidate_pages[base + index]);
    hash *= 1099511628211ull;
  }
  query_hashes[query] = hash;
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

cudaError_t fill_page_level_candidates(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    const float *descriptors, const float *queries,
    uint32_t *page_level_candidate_pages, cudaStream_t stream,
    NervaCudaExperimentalRtCandidateBenchResult *out) {
  page_level_candidate_selector_kernel<<<request->query_count, kThreads, 0,
                                         stream>>>(
      descriptors, queries, page_level_candidate_pages, request->pages,
      request->dims, request->query_count, request->candidates_per_query);
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) {
    out->kernel_launches += 1;
  }
  return err;
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

cudaError_t time_local_attention(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    const float *keys, const float *values, const float *queries,
    uint64_t local_window_tokens, float *local_out, float *local_meta,
    cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop,
    uint64_t *elapsed) {
  for (uint32_t iter = 0; iter < request->warmup_iterations; ++iter) {
    local_attention_kernel<<<request->query_count, kThreads, 0, stream>>>(
        keys, values, queries, request->pages, request->page_tokens,
        request->dims, request->query_count, local_window_tokens, local_out,
        local_meta);
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
    local_attention_kernel<<<request->query_count, kThreads, 0, stream>>>(
        keys, values, queries, request->pages, request->page_tokens,
        request->dims, request->query_count, local_window_tokens, local_out,
        local_meta);
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

cudaError_t time_dense_full_attention(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    const float *keys, const float *values, const float *queries,
    float *dense_out, cudaStream_t stream, cudaEvent_t start,
    cudaEvent_t stop, uint64_t *elapsed) {
  const uint64_t total_tokens =
      static_cast<uint64_t>(request->pages) * request->page_tokens;
  return time_local_attention(request, keys, values, queries, total_tokens,
                              dense_out, nullptr, stream, start, stop,
                              elapsed);
}

cudaError_t time_kv_page_access(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    const float *keys, const float *values, const uint32_t *candidate_pages,
    float *touch_out, cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop,
    uint64_t *elapsed) {
  for (uint32_t iter = 0; iter < request->warmup_iterations; ++iter) {
    kv_page_access_kernel<<<request->query_count, kThreads, 0, stream>>>(
        keys, values, candidate_pages, request->pages, request->page_tokens,
        request->dims, request->query_count, request->candidates_per_query,
        touch_out);
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
    kv_page_access_kernel<<<request->query_count, kThreads, 0, stream>>>(
        keys, values, candidate_pages, request->pages, request->page_tokens,
        request->dims, request->query_count, request->candidates_per_query,
        touch_out);
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

cudaError_t time_far_sparse_attention(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    const float *keys, const float *values, const float *queries,
    const uint32_t *candidate_pages, uint64_t local_window_tokens,
    float *far_out, float *far_meta, cudaStream_t stream, cudaEvent_t start,
    cudaEvent_t stop, uint64_t *elapsed) {
  for (uint32_t iter = 0; iter < request->warmup_iterations; ++iter) {
    far_sparse_attention_kernel<<<request->query_count, kThreads, 0, stream>>>(
        keys, values, queries, candidate_pages, request->pages,
        request->page_tokens, request->dims, request->query_count,
        request->candidates_per_query, local_window_tokens, far_out, far_meta);
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
    far_sparse_attention_kernel<<<request->query_count, kThreads, 0, stream>>>(
        keys, values, queries, candidate_pages, request->pages,
        request->page_tokens, request->dims, request->query_count,
        request->candidates_per_query, local_window_tokens, far_out, far_meta);
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

cudaError_t time_softmax_merge(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    const float *local_out, const float *local_meta, const float *far_out,
    const float *far_meta, float *merged_out, cudaStream_t stream,
    cudaEvent_t start, cudaEvent_t stop, uint64_t *elapsed) {
  const uint64_t total =
      static_cast<uint64_t>(request->query_count) * request->dims;
  const uint32_t blocks =
      static_cast<uint32_t>((total + kThreads - 1u) / kThreads);
  for (uint32_t iter = 0; iter < request->warmup_iterations; ++iter) {
    merge_attention_outputs_kernel<<<blocks, kThreads, 0, stream>>>(
        local_out, local_meta, far_out, far_meta, request->dims,
        request->query_count, merged_out);
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
    merge_attention_outputs_kernel<<<blocks, kThreads, 0, stream>>>(
        local_out, local_meta, far_out, far_meta, request->dims,
        request->query_count, merged_out);
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

uint32_t distinct_hash_count(const uint64_t *hashes, uint32_t count) {
  uint32_t distinct = 0;
  for (uint32_t index = 0; index < count; ++index) {
    bool seen = false;
    for (uint32_t previous = 0; previous < index; ++previous) {
      if (hashes[previous] == hashes[index]) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      ++distinct;
    }
  }
  return distinct;
}

cudaError_t verify_candidate_parity(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    const uint32_t *candidate_pages, cudaStream_t stream,
    NervaCudaExperimentalRtCandidateBenchResult *out) {
  uint32_t *software_candidate_pages = nullptr;
  unsigned long long *stats = nullptr;
  unsigned long long host_stats[4] = {};
  const uint64_t total_candidates =
      static_cast<uint64_t>(request->query_count) * request->candidates_per_query;
  cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&software_candidate_pages),
                               out->candidate_id_bytes);
  if (err == cudaSuccess) {
    out->device_allocations += 1;
    out->device_arena_bytes += out->candidate_id_bytes;
    err = cudaMalloc(reinterpret_cast<void **>(&stats),
                     sizeof(unsigned long long) * 4u);
    if (err == cudaSuccess) {
      out->device_allocations += 1;
      out->device_arena_bytes += sizeof(unsigned long long) * 4u;
    }
  }
  if (err != cudaSuccess) {
    if (software_candidate_pages != nullptr) {
      cudaFree(software_candidate_pages);
      out->device_frees += 1;
    }
    return err;
  }

  err = cudaMemsetAsync(stats, 0, sizeof(unsigned long long) * 4u, stream);
  if (err == cudaSuccess) {
    software_candidate_selector_kernel<<<request->query_count, kThreads, 0, stream>>>(
        software_candidate_pages, request->pages, request->query_count,
        request->candidates_per_query);
    const uint32_t blocks = static_cast<uint32_t>(
        (total_candidates + kThreads - 1u) / kThreads);
    compare_candidate_pages_kernel<<<blocks > kMaxInitBlocks ? kMaxInitBlocks : blocks,
                                     kThreads, 0, stream>>>(
        candidate_pages, software_candidate_pages, total_candidates, stats);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_stats, stats, sizeof(host_stats),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }

  cudaFree(stats);
  out->device_frees += 1;
  cudaFree(software_candidate_pages);
  out->device_frees += 1;
  if (err != cudaSuccess) {
    return err;
  }
  out->candidate_parity_checked = 1;
  out->candidate_parity_mismatches = host_stats[0];
  out->candidate_parity_first_mismatch_index = host_stats[1];
  out->candidate_parity_first_expected = host_stats[2];
  out->candidate_parity_first_actual = host_stats[3];
  out->kernel_launches += 2;
  out->sync_calls += 1;
  return cudaSuccess;
}

cudaError_t measure_query_candidate_distinctness(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    const uint32_t *candidate_pages, cudaStream_t stream,
    NervaCudaExperimentalRtCandidateBenchResult *out) {
  uint64_t *query_hashes = nullptr;
  uint64_t *host_query_hashes = nullptr;
  const uint64_t query_hash_bytes =
      static_cast<uint64_t>(request->query_count) * sizeof(uint64_t);
  cudaError_t err =
      cudaMalloc(reinterpret_cast<void **>(&query_hashes), query_hash_bytes);
  if (err == cudaSuccess) {
    out->device_allocations += 1;
    out->device_arena_bytes += query_hash_bytes;
  }
  if (err == cudaSuccess) {
    host_query_hashes = new (std::nothrow) uint64_t[request->query_count];
    if (host_query_hashes == nullptr) {
      err = cudaErrorMemoryAllocation;
    }
  }
  if (err == cudaSuccess) {
    const uint32_t blocks = (request->query_count + kThreads - 1u) / kThreads;
    hash_candidate_queries_kernel<<<blocks, kThreads, 0, stream>>>(
        candidate_pages, request->query_count, request->candidates_per_query,
        query_hashes);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_query_hashes, query_hashes, query_hash_bytes,
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (query_hashes != nullptr) {
    cudaFree(query_hashes);
    out->device_frees += 1;
  }
  if (err == cudaSuccess) {
    const uint32_t distinct =
        distinct_hash_count(host_query_hashes, request->query_count);
    out->candidate_query_hashes_distinct = distinct;
    out->candidate_query_hash_repeats = request->query_count - distinct;
    out->kernel_launches += 1;
    out->sync_calls += 1;
  }
  delete[] host_query_hashes;
  return err;
}

cudaError_t measure_attention_mass_recall(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    const float *keys, const float *queries, const uint32_t *candidate_pages,
    uint64_t local_window_tokens, uint64_t *recall_ppm,
    uint64_t *host_recall_ppm, uint64_t *min_ppm_out,
    uint64_t *avg_ppm_out, cudaStream_t stream,
    NervaCudaExperimentalRtCandidateBenchResult *out) {
  attention_mass_recall_kernel<<<request->query_count, kThreads, 0, stream>>>(
      keys, queries, candidate_pages, request->pages, request->page_tokens,
      request->dims, request->query_count, request->candidates_per_query,
      local_window_tokens, recall_ppm);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_recall_ppm, recall_ppm,
                          request->query_count * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (err != cudaSuccess) {
    return err;
  }

  uint64_t min_ppm = UINT64_MAX;
  uint64_t sum_ppm = 0;
  for (uint32_t query = 0; query < request->query_count; ++query) {
    const uint64_t value = host_recall_ppm[query];
    if (value < min_ppm) {
      min_ppm = value;
    }
    sum_ppm += value;
  }
  *min_ppm_out = min_ppm == UINT64_MAX ? 0 : min_ppm;
  *avg_ppm_out = request->query_count == 0 ? 0 : sum_ppm / request->query_count;
  out->kernel_launches += 1;
  out->sync_calls += 1;
  return cudaSuccess;
}

cudaError_t measure_far_oracle_topk_diagnostics(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    const float *keys, const float *queries, const uint32_t *candidate_pages,
    uint64_t local_window_tokens, uint64_t *token_recall_ppm,
    uint64_t *host_token_recall_ppm, uint64_t *token_recall_min_ppm_out,
    uint64_t *token_recall_avg_ppm_out, uint32_t *scatter_pages,
    uint32_t *host_scatter_pages, cudaStream_t stream,
    NervaCudaExperimentalRtCandidateBenchResult *out) {
  const uint64_t total_tokens =
      static_cast<uint64_t>(request->pages) * request->page_tokens;
  const uint64_t local_tokens =
      total_tokens < local_window_tokens ? total_tokens : local_window_tokens;
  const uint64_t far_tokens = total_tokens - local_tokens;
  uint64_t topk_tokens = request->candidates_per_query;
  if (topk_tokens > kMaxOracleTopK) {
    topk_tokens = kMaxOracleTopK;
  }
  if (topk_tokens > far_tokens) {
    topk_tokens = far_tokens;
  }
  out->far_oracle_topk_tokens = topk_tokens;
  if (topk_tokens == 0) {
    *token_recall_min_ppm_out = 0;
    *token_recall_avg_ppm_out = 0;
    out->far_oracle_topk_importance_scatter_min_pages = 0;
    out->far_oracle_topk_importance_scatter_avg_pages_x1000 = 0;
    out->far_oracle_topk_importance_scatter_max_pages = 0;
    return cudaSuccess;
  }

  far_oracle_topk_diagnostics_kernel<<<request->query_count, kThreads, 0,
                                       stream>>>(
      keys, queries, candidate_pages, request->pages, request->page_tokens,
      request->dims, request->query_count, request->candidates_per_query,
      local_window_tokens, static_cast<uint32_t>(topk_tokens),
      token_recall_ppm, scatter_pages);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_scatter_pages, scatter_pages,
                          request->query_count * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_token_recall_ppm, token_recall_ppm,
                          request->query_count * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (err != cudaSuccess) {
    return err;
  }

  uint64_t min_pages = UINT64_MAX;
  uint64_t sum_pages = 0;
  uint64_t max_pages = 0;
  uint64_t min_recall_ppm = UINT64_MAX;
  uint64_t sum_recall_ppm = 0;
  for (uint32_t query = 0; query < request->query_count; ++query) {
    const uint64_t value = host_scatter_pages[query];
    if (value < min_pages) {
      min_pages = value;
    }
    if (value > max_pages) {
      max_pages = value;
    }
    sum_pages += value;
    const uint64_t recall = host_token_recall_ppm[query];
    if (recall < min_recall_ppm) {
      min_recall_ppm = recall;
    }
    sum_recall_ppm += recall;
  }
  *token_recall_min_ppm_out =
      min_recall_ppm == UINT64_MAX ? 0 : min_recall_ppm;
  *token_recall_avg_ppm_out =
      request->query_count == 0 ? 0 : sum_recall_ppm / request->query_count;
  out->far_oracle_topk_importance_scatter_min_pages =
      min_pages == UINT64_MAX ? 0 : min_pages;
  out->far_oracle_topk_importance_scatter_avg_pages_x1000 =
      request->query_count == 0 ? 0
                                : (sum_pages * 1000ull) / request->query_count;
  out->far_oracle_topk_importance_scatter_max_pages = max_pages;
  out->kernel_launches += 1;
  out->sync_calls += 1;
  return cudaSuccess;
}

cudaError_t measure_fine_token_projected_topk_diagnostics(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    const float *keys, const float *queries, uint64_t local_window_tokens,
    uint64_t *token_recall_ppm, uint64_t *host_token_recall_ppm,
    uint32_t projection_dims, cudaStream_t stream,
    uint64_t *out_topk_tokens, uint64_t *out_candidate_tokens,
    uint64_t *out_token_recall_min_ppm,
    uint64_t *out_token_recall_avg_ppm,
    NervaCudaExperimentalRtCandidateBenchResult *out) {
  const uint64_t total_tokens =
      static_cast<uint64_t>(request->pages) * request->page_tokens;
  const uint64_t local_tokens =
      total_tokens < local_window_tokens ? total_tokens : local_window_tokens;
  const uint64_t far_tokens = total_tokens - local_tokens;
  uint64_t topk_tokens = request->candidates_per_query;
  if (topk_tokens > kMaxOracleTopK) {
    topk_tokens = kMaxOracleTopK;
  }
  if (topk_tokens > far_tokens) {
    topk_tokens = far_tokens;
  }
  uint64_t candidate_tokens = request->candidates_per_query;
  if (candidate_tokens > kMaxOracleTopK) {
    candidate_tokens = kMaxOracleTopK;
  }
  if (candidate_tokens > far_tokens) {
    candidate_tokens = far_tokens;
  }
  *out_topk_tokens = topk_tokens;
  *out_candidate_tokens = candidate_tokens;
  if (topk_tokens == 0 || candidate_tokens == 0) {
    *out_token_recall_min_ppm = 0;
    *out_token_recall_avg_ppm = 0;
    return cudaSuccess;
  }

  fine_token_projected_topk_diagnostics_kernel<<<request->query_count,
                                                  kThreads, 0, stream>>>(
      keys, queries, request->pages, request->page_tokens, request->dims,
      request->query_count, local_window_tokens,
      static_cast<uint32_t>(topk_tokens),
      static_cast<uint32_t>(candidate_tokens), projection_dims,
      token_recall_ppm);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_token_recall_ppm, token_recall_ppm,
                          request->query_count * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (err != cudaSuccess) {
    return err;
  }

  uint64_t min_recall_ppm = UINT64_MAX;
  uint64_t sum_recall_ppm = 0;
  for (uint32_t query = 0; query < request->query_count; ++query) {
    const uint64_t recall = host_token_recall_ppm[query];
    if (recall < min_recall_ppm) {
      min_recall_ppm = recall;
    }
    sum_recall_ppm += recall;
  }
  *out_token_recall_min_ppm = min_recall_ppm == UINT64_MAX ? 0 : min_recall_ppm;
  *out_token_recall_avg_ppm =
      request->query_count == 0 ? 0 : sum_recall_ppm / request->query_count;
  out->kernel_launches += 1;
  out->sync_calls += 1;
  return cudaSuccess;
}

cudaError_t measure_norm_stress_topk_diagnostics(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    const float *keys, const float *queries, const uint32_t *candidate_pages,
    uint64_t local_window_tokens, uint64_t *no_augmentation_recall_ppm,
    uint64_t *host_no_augmentation_recall_ppm,
    uint64_t *synthetic_norm_augmented_recall_ppm,
    uint64_t *host_synthetic_norm_augmented_recall_ppm, cudaStream_t stream,
    NervaCudaExperimentalRtCandidateBenchResult *out) {
  const uint64_t total_tokens =
      static_cast<uint64_t>(request->pages) * request->page_tokens;
  const uint64_t local_tokens =
      total_tokens < local_window_tokens ? total_tokens : local_window_tokens;
  const uint64_t far_tokens = total_tokens - local_tokens;
  uint64_t topk_tokens = request->candidates_per_query;
  if (topk_tokens > kMaxOracleTopK) {
    topk_tokens = kMaxOracleTopK;
  }
  if (topk_tokens > far_tokens) {
    topk_tokens = far_tokens;
  }
  out->norm_stress_topk_tokens = topk_tokens;
  if (topk_tokens == 0) {
    out->norm_stress_no_augmentation_token_recall_min_ppm = 0;
    out->norm_stress_no_augmentation_token_recall_avg_ppm = 0;
    out->norm_stress_synthetic_norm_augmented_token_recall_min_ppm = 0;
    out->norm_stress_synthetic_norm_augmented_token_recall_avg_ppm = 0;
    return cudaSuccess;
  }

  norm_stress_topk_diagnostics_kernel<<<request->query_count, kThreads, 0,
                                        stream>>>(
      keys, queries, candidate_pages, request->pages, request->page_tokens,
      request->dims, request->query_count, request->candidates_per_query,
      local_window_tokens, static_cast<uint32_t>(topk_tokens),
      no_augmentation_recall_ppm, synthetic_norm_augmented_recall_ppm);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_no_augmentation_recall_ppm,
                          no_augmentation_recall_ppm,
                          request->query_count * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_synthetic_norm_augmented_recall_ppm,
                          synthetic_norm_augmented_recall_ppm,
                          request->query_count * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (err != cudaSuccess) {
    return err;
  }

  uint64_t no_aug_min = UINT64_MAX;
  uint64_t no_aug_sum = 0;
  uint64_t norm_aug_min = UINT64_MAX;
  uint64_t norm_aug_sum = 0;
  for (uint32_t query = 0; query < request->query_count; ++query) {
    const uint64_t no_aug = host_no_augmentation_recall_ppm[query];
    const uint64_t norm_aug = host_synthetic_norm_augmented_recall_ppm[query];
    if (no_aug < no_aug_min) {
      no_aug_min = no_aug;
    }
    if (norm_aug < norm_aug_min) {
      norm_aug_min = norm_aug;
    }
    no_aug_sum += no_aug;
    norm_aug_sum += norm_aug;
  }
  out->norm_stress_no_augmentation_token_recall_min_ppm =
      no_aug_min == UINT64_MAX ? 0 : no_aug_min;
  out->norm_stress_no_augmentation_token_recall_avg_ppm =
      request->query_count == 0 ? 0 : no_aug_sum / request->query_count;
  out->norm_stress_synthetic_norm_augmented_token_recall_min_ppm =
      norm_aug_min == UINT64_MAX ? 0 : norm_aug_min;
  out->norm_stress_synthetic_norm_augmented_token_recall_avg_ppm =
      request->query_count == 0 ? 0 : norm_aug_sum / request->query_count;
  out->kernel_launches += 1;
  out->sync_calls += 1;
  return cudaSuccess;
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
  out->kv_cache_bytes = checked_mul_u64(
      checked_mul_u64(checked_mul_u64(request->pages, request->page_tokens),
                      request->dims),
      sizeof(float) * 2ull);
  out->candidate_id_bytes = checked_mul_u64(
      checked_mul_u64(request->query_count, request->candidates_per_query), sizeof(uint32_t));
  out->output_bytes = checked_mul_u64(request->query_count, sizeof(uint32_t)) * 2ull;
  out->device_arena_bytes = out->descriptor_bytes + out->query_bytes +
                            out->kv_cache_bytes + out->candidate_id_bytes +
                            out->output_bytes;
}

int fail(NervaCudaExperimentalRtCandidateBenchResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

}  // namespace

struct NervaCudaRtCandidateSelectorHandle {
  NervaCudaExperimentalRtCandidateBenchRequest request{};
  NervaCudaExperimentalRtCandidateBenchResult result{};
  uint32_t *candidate_pages = nullptr;
#if NERVA_HAVE_OPTIX_HEADERS
  OptixCandidateSelector selector{};
#endif
};

int nerva_cuda_rt_candidate_selector_create_impl(
    uint32_t pages, uint32_t page_tokens, uint32_t query_count,
    uint32_t candidates_per_query, uint32_t *candidate_pages,
    const float *queries, uint32_t query_dims, const uint32_t *step_cursor,
    const float *page_descriptors, uint32_t page_descriptor_dims,
    uint32_t layer_count,
    void *stream, void **selector_out, int32_t *cuda_error_out) {
  if (cuda_error_out != nullptr) {
    *cuda_error_out = static_cast<int32_t>(cudaSuccess);
  }
  if (selector_out == nullptr) {
    if (cuda_error_out != nullptr) {
      *cuda_error_out = static_cast<int32_t>(cudaErrorInvalidValue);
    }
    return -1;
  }
  *selector_out = nullptr;
  if (pages == 0 || page_tokens == 0 || query_count == 0 ||
      candidates_per_query == 0 || candidates_per_query > pages ||
      candidate_pages == nullptr || stream == nullptr) {
    if (cuda_error_out != nullptr) {
      *cuda_error_out = static_cast<int32_t>(cudaErrorInvalidValue);
    }
    return -1;
  }
#if NERVA_HAVE_OPTIX_HEADERS
  auto *handle = new (std::nothrow) NervaCudaRtCandidateSelectorHandle();
  if (handle == nullptr) {
    if (cuda_error_out != nullptr) {
      *cuda_error_out = static_cast<int32_t>(cudaErrorMemoryAllocation);
    }
    return -1;
  }
  handle->request.pages = pages;
  handle->request.page_tokens = page_tokens;
  handle->request.dims = 1;
  handle->request.query_count = query_count;
  handle->request.candidates_per_query = candidates_per_query;
  handle->request.iterations = 1;
  handle->request.warmup_iterations = 0;
  handle->candidate_pages = candidate_pages;
  clear_result(&handle->request, &handle->result);
  cudaError_t err = cudaGetDeviceCount(&handle->result.device_count);
  if (err == cudaSuccess && handle->result.device_count <= 0) {
    err = cudaErrorNoDevice;
  }
  if (err == cudaSuccess) {
    err = cudaGetDevice(&handle->result.device_ordinal);
  }
  cudaDeviceProp props{};
  if (err == cudaSuccess) {
    err = cudaGetDeviceProperties(&props, handle->result.device_ordinal);
  }
  if (err != cudaSuccess) {
    if (cuda_error_out != nullptr) {
      *cuda_error_out = static_cast<int32_t>(err);
    }
    delete handle;
    return -1;
  }
  handle->result.compute_capability_major = props.major;
  handle->result.compute_capability_minor = props.minor;
  handle->result.rt_core_capable = props.major >= 7 ? 1u : 0u;
  handle->result.optix_headers_available = NERVA_HAVE_OPTIX_HEADERS ? 1u : 0u;
  handle->result.rt_headers_available = 1u;
  handle->result.real_rt_backend_available = 0u;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (!create_optix_candidate_selector(&handle->request, candidate_pages,
                                       cuda_stream, &handle->selector,
                                       &handle->result, queries, query_dims,
                                       queries == nullptr ? 0u : 1u,
                                       step_cursor, page_descriptors,
                                       page_descriptor_dims, layer_count)) {
    cleanup_optix_selector(&handle->selector, &handle->result);
    if (cuda_error_out != nullptr) {
      *cuda_error_out =
          handle->result.cuda_error == 0
              ? static_cast<int32_t>(cudaErrorNotSupported)
              : handle->result.cuda_error;
    }
    delete handle;
    return -1;
  }
  handle->result.real_rt_backend_available = 1u;
  *selector_out = handle;
  return 0;
#else
  if (cuda_error_out != nullptr) {
    *cuda_error_out = static_cast<int32_t>(cudaErrorNotSupported);
  }
  return -1;
#endif
}

extern "C" int nerva_cuda_rt_candidate_selector_create(
    uint32_t pages, uint32_t page_tokens, uint32_t query_count,
    uint32_t candidates_per_query, uint32_t *candidate_pages, void *stream,
    void **selector_out, int32_t *cuda_error_out) {
  return nerva_cuda_rt_candidate_selector_create_impl(
      pages, page_tokens, query_count, candidates_per_query, candidate_pages,
      nullptr, 0, nullptr, nullptr, 0, 1u, stream, selector_out,
      cuda_error_out);
}

extern "C" int nerva_cuda_rt_candidate_selector_create_with_queries(
    uint32_t pages, uint32_t page_tokens, uint32_t query_count,
    uint32_t candidates_per_query, uint32_t *candidate_pages,
    const float *queries, uint32_t query_dims, const uint32_t *step_cursor,
    void *stream, void **selector_out, int32_t *cuda_error_out) {
  return nerva_cuda_rt_candidate_selector_create_impl(
      pages, page_tokens, query_count, candidates_per_query, candidate_pages,
      queries, query_dims, step_cursor, nullptr, 0, 1u, stream, selector_out,
      cuda_error_out);
}

extern "C" int
nerva_cuda_rt_candidate_selector_create_with_query_page_descriptors(
    uint32_t pages, uint32_t page_tokens, uint32_t layer_count,
    uint32_t query_count, uint32_t candidates_per_query,
    uint32_t *candidate_pages, const float *queries, uint32_t query_dims,
    const float *page_descriptors, uint32_t page_descriptor_dims,
    const uint32_t *step_cursor, void *stream, void **selector_out,
    int32_t *cuda_error_out) {
  return nerva_cuda_rt_candidate_selector_create_impl(
      pages, page_tokens, query_count, candidates_per_query, candidate_pages,
      queries, query_dims, step_cursor, page_descriptors, page_descriptor_dims,
      layer_count, stream, selector_out, cuda_error_out);
}

extern "C" int nerva_cuda_rt_candidate_selector_launch(
    void *selector, void *stream, uint32_t active_pages, uint32_t current_page,
    uint32_t local_pages, uint32_t sink_pages, uint32_t layer_index,
    int32_t *cuda_error_out) {
  if (cuda_error_out != nullptr) {
    *cuda_error_out = static_cast<int32_t>(cudaSuccess);
  }
  if (selector == nullptr || stream == nullptr) {
    if (cuda_error_out != nullptr) {
      *cuda_error_out = static_cast<int32_t>(cudaErrorInvalidValue);
    }
    return -1;
  }
#if NERVA_HAVE_OPTIX_HEADERS
  auto *handle =
      reinterpret_cast<NervaCudaRtCandidateSelectorHandle *>(selector);
  const uint32_t pages =
      active_pages == 0 || active_pages > handle->request.pages
          ? handle->request.pages
          : active_pages;
  OptixCandidateParams params{};
  params.handle = handle->selector.traversable;
  params.candidate_pages = handle->candidate_pages;
  params.queries = handle->selector.queries;
  params.step_cursor = handle->selector.step_cursor;
  params.pages = pages;
  params.query_count = handle->request.query_count;
  params.candidates_per_query = handle->request.candidates_per_query;
  params.layer_index =
      layer_index < handle->selector.layer_count ? layer_index : 0u;
  params.layer_count = handle->selector.layer_count;
  params.query_dims = handle->selector.query_dims;
  params.query_derived_pages = handle->selector.query_derived_pages;
  params.descriptor_geometry = handle->selector.descriptor_geometry;
  params.page_tokens_for_step = handle->request.page_tokens;
  params.dynamic_step = handle->selector.dynamic_step;
  params.grid_width = handle->selector.grid_width;
  params.current_page = current_page < pages ? current_page : pages - 1u;
  params.local_pages = local_pages;
  params.sink_pages = sink_pages;
  params.cell_size = handle->selector.cell_size;
  params.descriptor_scale = handle->selector.descriptor_scale;
  params.descriptor_plane_stride = handle->selector.descriptor_plane_stride;
  cudaError_t err = cudaMemcpyAsync(
      reinterpret_cast<void *>(handle->selector.params), &params,
      sizeof(params), cudaMemcpyHostToDevice,
      reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    if (cuda_error_out != nullptr) {
      *cuda_error_out = static_cast<int32_t>(err);
    }
    return -1;
  }
  OptixResult optix = optixLaunch(
      handle->selector.pipeline, reinterpret_cast<CUstream>(stream),
      handle->selector.params, sizeof(OptixCandidateParams),
      &handle->selector.sbt, handle->request.candidates_per_query,
      handle->request.query_count, 1);
  if (optix != OPTIX_SUCCESS) {
    if (cuda_error_out != nullptr) {
      *cuda_error_out = static_cast<int32_t>(cudaErrorUnknown);
    }
    return -1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (cuda_error_out != nullptr) {
      *cuda_error_out = static_cast<int32_t>(err);
    }
    return -1;
  }
  return 0;
#else
  if (cuda_error_out != nullptr) {
    *cuda_error_out = static_cast<int32_t>(cudaErrorNotSupported);
  }
  return -1;
#endif
}

extern "C" void nerva_cuda_rt_candidate_selector_destroy(void *selector) {
  if (selector == nullptr) {
    return;
  }
  auto *handle =
      reinterpret_cast<NervaCudaRtCandidateSelectorHandle *>(selector);
#if NERVA_HAVE_OPTIX_HEADERS
  cleanup_optix_selector(&handle->selector, &handle->result);
#endif
  delete handle;
}

extern "C" int nerva_cuda_experimental_rt_candidate_bench(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    NervaCudaExperimentalRtCandidateBenchResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(request, out);
  if (request == nullptr || request->pages == 0 || request->page_tokens == 0 ||
      request->dims == 0 || request->dims > kMaxAttentionDims || request->query_count == 0 ||
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
  float *keys = nullptr;
  float *values = nullptr;
  float *local_attention_out = nullptr;
  float *local_attention_meta = nullptr;
  float *far_attention_out = nullptr;
  float *far_attention_meta = nullptr;
  float *merged_attention_out = nullptr;
  float *dense_attention_out = nullptr;
  float *kv_touch_out = nullptr;
  uint64_t *attention_recall_ppm = nullptr;
  uint64_t *norm_augmented_recall_ppm = nullptr;
  uint32_t *far_oracle_scatter_pages = nullptr;
  uint32_t *candidate_pages = nullptr;
  uint32_t *page_level_candidate_pages = nullptr;
  uint32_t *dense_out = nullptr;
  uint32_t *candidate_out = nullptr;
  uint32_t *host_selected = nullptr;
  uint64_t *host_attention_recall_ppm = nullptr;
  uint64_t *host_norm_augmented_recall_ppm = nullptr;
  uint32_t *host_far_oracle_scatter_pages = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  auto cleanup = [&]() {
    if (host_attention_recall_ppm != nullptr) {
      delete[] host_attention_recall_ppm;
    }
    if (host_norm_augmented_recall_ppm != nullptr) {
      delete[] host_norm_augmented_recall_ppm;
    }
    if (host_far_oracle_scatter_pages != nullptr) {
      delete[] host_far_oracle_scatter_pages;
    }
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
    if (kv_touch_out != nullptr) {
      cudaFree(kv_touch_out);
      out->device_frees += 1;
    }
    if (attention_recall_ppm != nullptr) {
      cudaFree(attention_recall_ppm);
      out->device_frees += 1;
    }
    if (norm_augmented_recall_ppm != nullptr) {
      cudaFree(norm_augmented_recall_ppm);
      out->device_frees += 1;
    }
    if (far_oracle_scatter_pages != nullptr) {
      cudaFree(far_oracle_scatter_pages);
      out->device_frees += 1;
    }
    if (dense_attention_out != nullptr) {
      cudaFree(dense_attention_out);
      out->device_frees += 1;
    }
    if (far_attention_meta != nullptr) {
      cudaFree(far_attention_meta);
      out->device_frees += 1;
    }
    if (merged_attention_out != nullptr) {
      cudaFree(merged_attention_out);
      out->device_frees += 1;
    }
    if (far_attention_out != nullptr) {
      cudaFree(far_attention_out);
      out->device_frees += 1;
    }
    if (local_attention_meta != nullptr) {
      cudaFree(local_attention_meta);
      out->device_frees += 1;
    }
    if (local_attention_out != nullptr) {
      cudaFree(local_attention_out);
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
    if (page_level_candidate_pages != nullptr) {
      cudaFree(page_level_candidate_pages);
      out->device_frees += 1;
    }
    if (queries != nullptr) {
      cudaFree(queries);
      out->device_frees += 1;
    }
    if (values != nullptr) {
      cudaFree(values);
      out->device_frees += 1;
    }
    if (keys != nullptr) {
      cudaFree(keys);
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
  const uint64_t single_kv_bytes = out->kv_cache_bytes / 2ull;
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&keys), single_kv_bytes);
    if (err == cudaSuccess) out->device_allocations += 1;
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&values), single_kv_bytes);
    if (err == cudaSuccess) out->device_allocations += 1;
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&candidate_pages), out->candidate_id_bytes);
    if (err == cudaSuccess) out->device_allocations += 1;
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&page_level_candidate_pages),
                     out->candidate_id_bytes);
    if (err == cudaSuccess) {
      out->device_allocations += 1;
      out->device_arena_bytes += out->candidate_id_bytes;
    }
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
  const uint64_t attention_output_bytes =
      checked_mul_u64(checked_mul_u64(request->query_count, request->dims),
                      sizeof(float));
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&local_attention_out),
                     attention_output_bytes);
    if (err == cudaSuccess) {
      out->device_allocations += 1;
      out->device_arena_bytes += attention_output_bytes;
    }
  }
  const uint64_t attention_meta_bytes =
      checked_mul_u64(request->query_count, sizeof(float) * 2ull);
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&local_attention_meta),
                     attention_meta_bytes);
    if (err == cudaSuccess) {
      out->device_allocations += 1;
      out->device_arena_bytes += attention_meta_bytes;
    }
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&far_attention_out),
                     attention_output_bytes);
    if (err == cudaSuccess) {
      out->device_allocations += 1;
      out->device_arena_bytes += attention_output_bytes;
    }
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&far_attention_meta),
                     attention_meta_bytes);
    if (err == cudaSuccess) {
      out->device_allocations += 1;
      out->device_arena_bytes += attention_meta_bytes;
    }
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&merged_attention_out),
                     attention_output_bytes);
    if (err == cudaSuccess) {
      out->device_allocations += 1;
      out->device_arena_bytes += attention_output_bytes;
    }
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&dense_attention_out),
                     attention_output_bytes);
    if (err == cudaSuccess) {
      out->device_allocations += 1;
      out->device_arena_bytes += attention_output_bytes;
    }
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&attention_recall_ppm),
                     request->query_count * sizeof(uint64_t));
    if (err == cudaSuccess) {
      out->device_allocations += 1;
      out->device_arena_bytes += request->query_count * sizeof(uint64_t);
    }
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&norm_augmented_recall_ppm),
                     request->query_count * sizeof(uint64_t));
    if (err == cudaSuccess) {
      out->device_allocations += 1;
      out->device_arena_bytes += request->query_count * sizeof(uint64_t);
    }
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&far_oracle_scatter_pages),
                     request->query_count * sizeof(uint32_t));
    if (err == cudaSuccess) {
      out->device_allocations += 1;
      out->device_arena_bytes += request->query_count * sizeof(uint32_t);
    }
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&kv_touch_out),
                     request->query_count * sizeof(float));
    if (err == cudaSuccess) {
      out->device_allocations += 1;
      out->device_arena_bytes += request->query_count * sizeof(float);
    }
  }
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  host_selected = new (std::nothrow) uint32_t[request->query_count];
  if (host_selected == nullptr) {
    return fail_with_cleanup(cudaErrorMemoryAllocation);
  }
  host_attention_recall_ppm = new (std::nothrow) uint64_t[request->query_count];
  if (host_attention_recall_ppm == nullptr) {
    return fail_with_cleanup(cudaErrorMemoryAllocation);
  }
  host_norm_augmented_recall_ppm =
      new (std::nothrow) uint64_t[request->query_count];
  if (host_norm_augmented_recall_ppm == nullptr) {
    return fail_with_cleanup(cudaErrorMemoryAllocation);
  }
  host_far_oracle_scatter_pages =
      new (std::nothrow) uint32_t[request->query_count];
  if (host_far_oracle_scatter_pages == nullptr) {
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
                                                         request->pages,
                                                         request->dims,
                                                         request->candidates_per_query,
                                                         request->page_tokens);
  const uint64_t kv_elements = static_cast<uint64_t>(request->pages) *
                               request->page_tokens * request->dims;
  const uint32_t kv_blocks = static_cast<uint32_t>(
      kv_elements / kThreads + (kv_elements % kThreads != 0));
  init_kv_cache_kernel<<<kv_blocks > kMaxInitBlocks ? kMaxInitBlocks : kv_blocks,
                         kThreads, 0, stream>>>(
      keys, values, request->pages, request->page_tokens, request->dims);
  err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->kernel_launches += 3;
  out->sync_calls += 1;
  out->local_window_tokens =
      (static_cast<uint64_t>(request->pages) * request->page_tokens) <
              kDefaultLocalWindowTokens
          ? (static_cast<uint64_t>(request->pages) * request->page_tokens)
          : kDefaultLocalWindowTokens;

  err = time_dense_selector(request, descriptors, queries, dense_out, stream, start, stop,
                            &out->dense_selector_total_ns);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->sync_calls += 1;
  out->kernel_launches += request->warmup_iterations + request->iterations;

  bool used_optix_selector = false;
  const bool use_semantic_optix_selector =
      env_truthy("NERVA_EXPERIMENTAL_RT_SEMANTIC_OPTIX");
#if NERVA_HAVE_OPTIX_HEADERS
  if (out->optix_headers_available != 0) {
    OptixCandidateSelector optix_selector{};
    if (create_optix_candidate_selector(request, candidate_pages, stream,
                                        &optix_selector, out,
                                        use_semantic_optix_selector ? queries
                                                                    : nullptr,
                                        use_semantic_optix_selector
                                            ? request->dims
                                            : 0u,
                                        use_semantic_optix_selector ? 1u
                                                                    : 0u)) {
      used_optix_selector =
          time_optix_selector(request, &optix_selector, stream, start, stop,
                              &out->software_selector_total_ns, out);
      if (used_optix_selector) {
        if (use_semantic_optix_selector) {
          set_cstr(out->backend, sizeof(out->backend),
                   "optix_rt_query_descriptor_candidate_selector");
          set_cstr(out->reason, sizeof(out->reason),
                   "OptiX hardware traversal derived page candidates from "
                   "query descriptors; exact rerank remains CUDA");
        } else {
          set_cstr(out->backend, sizeof(out->backend),
                   "optix_rt_candidate_selector");
          set_cstr(out->reason, sizeof(out->reason),
                   "OptiX hardware traversal generated score-aligned synthetic "
                   "page candidates; exact rerank remains CUDA");
        }
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

  if (used_optix_selector) {
    err = verify_candidate_parity(request, candidate_pages, stream, out);
    if (err != cudaSuccess) {
      return fail_with_cleanup(err);
    }
  }
  err = measure_query_candidate_distinctness(request, candidate_pages, stream, out);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }

  err = time_rerank(request, descriptors, queries, candidate_pages, candidate_out, stream,
                    start, stop, &out->rerank_total_ns);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->sync_calls += 1;
  out->kernel_launches += request->warmup_iterations + request->iterations;

  err = time_local_attention(request, keys, values, queries,
                             out->local_window_tokens,
                             local_attention_out, local_attention_meta,
                             stream, start, stop,
                             &out->local_attention_total_ns);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->sync_calls += 1;
  out->kernel_launches += request->warmup_iterations + request->iterations;

  err = time_kv_page_access(request, keys, values, candidate_pages, kv_touch_out,
                            stream, start, stop,
                            &out->kv_page_access_total_ns);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->sync_calls += 1;
  out->kernel_launches += request->warmup_iterations + request->iterations;

  err = time_far_sparse_attention(request, keys, values, queries, candidate_pages,
                                  out->local_window_tokens, far_attention_out,
                                  far_attention_meta, stream, start, stop,
                                  &out->far_sparse_attention_total_ns);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->sync_calls += 1;
  out->kernel_launches += request->warmup_iterations + request->iterations;

  err = time_softmax_merge(request, local_attention_out, local_attention_meta,
                           far_attention_out, far_attention_meta,
                           merged_attention_out, stream, start, stop,
                           &out->softmax_merge_total_ns);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->sync_calls += 1;
  out->kernel_launches += request->warmup_iterations + request->iterations;

  err = time_dense_full_attention(request, keys, values, queries,
                                  dense_attention_out, stream, start, stop,
                                  &out->dense_full_attention_total_ns);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->sync_calls += 1;
  out->kernel_launches += request->warmup_iterations + request->iterations;

  err = measure_far_oracle_topk_diagnostics(
      request, keys, queries, candidate_pages, out->local_window_tokens,
      attention_recall_ppm, host_attention_recall_ppm,
      &out->far_oracle_topk_token_recall_min_ppm,
      &out->far_oracle_topk_token_recall_avg_ppm, far_oracle_scatter_pages,
      host_far_oracle_scatter_pages, stream, out);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }

  err = measure_fine_token_projected_topk_diagnostics(
      request, keys, queries, out->local_window_tokens, attention_recall_ppm,
      host_attention_recall_ppm, 3u, stream,
      &out->fine_token_projected_topk_tokens,
      &out->fine_token_projected_candidate_tokens,
      &out->fine_token_projected_token_recall_min_ppm,
      &out->fine_token_projected_token_recall_avg_ppm, out);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }

  err = measure_fine_token_projected_topk_diagnostics(
      request, keys, queries, out->local_window_tokens, attention_recall_ppm,
      host_attention_recall_ppm, 4u, stream,
      &out->fine_token_learned_projected_topk_tokens,
      &out->fine_token_learned_projected_candidate_tokens,
      &out->fine_token_learned_projected_token_recall_min_ppm,
      &out->fine_token_learned_projected_token_recall_avg_ppm, out);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }

  err = measure_norm_stress_topk_diagnostics(
      request, keys, queries, candidate_pages, out->local_window_tokens,
      attention_recall_ppm, host_attention_recall_ppm,
      norm_augmented_recall_ppm, host_norm_augmented_recall_ppm, stream, out);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }

  err = measure_attention_mass_recall(
      request, keys, queries, candidate_pages, out->local_window_tokens,
      attention_recall_ppm, host_attention_recall_ppm,
      &out->attention_mass_recall_min_ppm,
      &out->attention_mass_recall_avg_ppm, stream, out);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }

  err = fill_page_level_candidates(request, descriptors, queries,
                                   page_level_candidate_pages, stream, out);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  err = measure_far_oracle_topk_diagnostics(
      request, keys, queries, page_level_candidate_pages, out->local_window_tokens,
      attention_recall_ppm, host_attention_recall_ppm,
      &out->page_level_far_oracle_topk_token_recall_min_ppm,
      &out->page_level_far_oracle_topk_token_recall_avg_ppm,
      far_oracle_scatter_pages, host_far_oracle_scatter_pages, stream, out);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  err = measure_attention_mass_recall(
      request, keys, queries, page_level_candidate_pages, out->local_window_tokens,
      attention_recall_ppm, host_attention_recall_ppm,
      &out->page_level_attention_mass_recall_min_ppm,
      &out->page_level_attention_mass_recall_avg_ppm, stream, out);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }

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
  out->candidate_selector_total_ns = out->software_selector_total_ns;
  out->candidate_selector_avg_ns = out->software_selector_avg_ns;
  out->rerank_avg_ns = out->rerank_total_ns / request->iterations;
  out->local_attention_avg_ns =
      out->local_attention_total_ns / request->iterations;
  out->kv_page_access_avg_ns =
      out->kv_page_access_total_ns / request->iterations;
  out->far_sparse_attention_avg_ns =
      out->far_sparse_attention_total_ns / request->iterations;
  out->softmax_merge_avg_ns =
      out->softmax_merge_total_ns / request->iterations;
  out->dense_full_attention_avg_ns =
      out->dense_full_attention_total_ns / request->iterations;
  out->selector_plus_rerank_avg_ns =
      out->candidate_selector_avg_ns + out->rerank_avg_ns;
  out->dense_selector_attention_stage_avg_ns =
      out->dense_selector_avg_ns + out->rerank_avg_ns +
      out->local_attention_avg_ns + out->kv_page_access_avg_ns +
      out->far_sparse_attention_avg_ns + out->softmax_merge_avg_ns;
  out->rt_selector_attention_stage_avg_ns =
      out->candidate_selector_avg_ns + out->rerank_avg_ns +
      out->local_attention_avg_ns + out->kv_page_access_avg_ns +
      out->far_sparse_attention_avg_ns + out->softmax_merge_avg_ns;
  const uint64_t overlapped_selector_local =
      out->candidate_selector_avg_ns > out->local_attention_avg_ns
          ? out->candidate_selector_avg_ns
          : out->local_attention_avg_ns;
  out->rt_selector_overlapped_attention_stage_avg_ns =
      overlapped_selector_local + out->rerank_avg_ns +
      out->kv_page_access_avg_ns + out->far_sparse_attention_avg_ns +
      out->softmax_merge_avg_ns;
  out->dense_vs_selector_speedup_x1000 =
      speedup_x1000(out->dense_selector_avg_ns, out->candidate_selector_avg_ns);
  out->dense_vs_selector_plus_rerank_speedup_x1000 =
      speedup_x1000(out->dense_selector_avg_ns, out->selector_plus_rerank_avg_ns);
  out->dense_vs_rt_attention_stage_speedup_x1000 =
      speedup_x1000(out->dense_selector_attention_stage_avg_ns,
                    out->rt_selector_attention_stage_avg_ns);
  out->dense_vs_rt_overlapped_attention_stage_speedup_x1000 =
      speedup_x1000(out->dense_selector_attention_stage_avg_ns,
                    out->rt_selector_overlapped_attention_stage_avg_ns);
  out->dense_full_vs_rt_attention_stage_speedup_x1000 =
      speedup_x1000(out->dense_full_attention_avg_ns,
                    out->rt_selector_attention_stage_avg_ns);
  out->dense_full_vs_rt_overlapped_attention_stage_speedup_x1000 =
      speedup_x1000(out->dense_full_attention_avg_ns,
                    out->rt_selector_overlapped_attention_stage_avg_ns);
  out->candidate_fraction_ppm =
      div_u64(static_cast<uint64_t>(request->candidates_per_query) * 1000000ull,
              request->pages);
  out->hot_path_allocations = 0;
  out->status = 0;

  cleanup();
  return 0;
}

namespace {

void clear_cold_kv_result(
    const NervaCudaExperimentalRtColdKvStagingRequest *request,
    NervaCudaExperimentalRtColdKvStagingResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->device_ordinal = -1;
  set_cstr(out->backend, sizeof(out->backend),
           "cuda_pinned_h2d_cold_kv_staging");
  set_cstr(out->reason, sizeof(out->reason),
           "Pinned host to device cold KV page staging benchmark");
  if (request == nullptr) {
    return;
  }
  out->page_bytes = request->page_bytes;
  out->pages_per_step = request->pages_per_step;
  out->iterations = request->iterations;
  out->warmup_iterations = request->warmup_iterations;
  out->bytes_per_step =
      checked_mul_u64(request->page_bytes, request->pages_per_step);
  out->total_h2d_bytes =
      checked_mul_u64(out->bytes_per_step, request->iterations);
}

int fail_cold_kv(NervaCudaExperimentalRtColdKvStagingResult *out,
                 cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

}  // namespace

extern "C" int nerva_cuda_experimental_rt_cold_kv_staging_bench(
    const NervaCudaExperimentalRtColdKvStagingRequest *request,
    NervaCudaExperimentalRtColdKvStagingResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_cold_kv_result(request, out);
  if (request == nullptr || request->page_bytes == 0 ||
      request->pages_per_step == 0 || request->iterations == 0 ||
      out->bytes_per_step == 0 || out->bytes_per_step == UINT64_MAX ||
      out->total_h2d_bytes == UINT64_MAX) {
    return fail_cold_kv(out, cudaErrorInvalidValue);
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_cold_kv(out, err);
  }
  if (out->device_count <= 0) {
    return fail_cold_kv(out, cudaErrorNoDevice);
  }
  err = cudaGetDevice(&out->device_ordinal);
  if (err != cudaSuccess) {
    return fail_cold_kv(out, err);
  }
  cudaDeviceProp props{};
  err = cudaGetDeviceProperties(&props, out->device_ordinal);
  if (err != cudaSuccess) {
    return fail_cold_kv(out, err);
  }
  out->compute_capability_major = props.major;
  out->compute_capability_minor = props.minor;

  void *host = nullptr;
  void *device = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  auto cleanup = [&]() {
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    if (device != nullptr) {
      cudaFree(device);
      out->device_frees += 1;
    }
    if (host != nullptr) {
      cudaFreeHost(host);
      out->pinned_host_frees += 1;
    }
  };
  auto fail_with_cleanup = [&](cudaError_t failure) {
    cleanup();
    return fail_cold_kv(out, failure);
  };

  const size_t bytes_per_step = static_cast<size_t>(out->bytes_per_step);
  if (static_cast<uint64_t>(bytes_per_step) != out->bytes_per_step) {
    return fail_cold_kv(out, cudaErrorInvalidValue);
  }
  err = cudaHostAlloc(&host, bytes_per_step, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    return fail_cold_kv(out, err);
  }
  out->pinned_host_allocations += 1;
  out->pinned_host_bytes = out->bytes_per_step;
  memset(host, 0x5a, bytes_per_step);

  err = cudaMalloc(&device, bytes_per_step);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  out->device_allocations += 1;
  out->device_arena_bytes = out->bytes_per_step;

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

  for (uint32_t iter = 0; iter < request->warmup_iterations; ++iter) {
    err = cudaMemcpyAsync(device, host, bytes_per_step, cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) {
      return fail_with_cleanup(err);
    }
  }
  err = cudaEventRecord(start, stream);
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }
  for (uint32_t iter = 0; iter < request->iterations; ++iter) {
    err = cudaMemcpyAsync(device, host, bytes_per_step, cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) {
      return fail_with_cleanup(err);
    }
  }
  err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) {
    err = cudaEventSynchronize(stop);
    out->sync_calls += 1;
  }
  if (err != cudaSuccess) {
    return fail_with_cleanup(err);
  }

  out->h2d_total_ns = elapsed_ns(start, stop);
  out->h2d_avg_ns = div_u64(out->h2d_total_ns, request->iterations);
  out->h2d_avg_page_ns = div_u64(out->h2d_avg_ns, request->pages_per_step);
  out->effective_bandwidth_bps =
      mul_div_u64(out->total_h2d_bytes, 1000000000ull, out->h2d_total_ns);
  out->hot_path_allocations = 0;
  out->status = 0;
  cleanup();
  return 0;
}
