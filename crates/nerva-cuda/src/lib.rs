//! # WARNING: CUDA 12/13 Only
//!
//! NERVA's CUDA path currently supports CUDA 12.x and CUDA 13.x only. Older
//! CUDA driver/runtime stacks are intentionally rejected instead of silently
//! falling back.

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;

type CuResult = c_int;
type CuDevice = c_int;
type CuDevicePtr = u64;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const CUDA_ERROR_INVALID_DEVICE: CuResult = 101;
const CUDA_ERROR_NO_DEVICE: CuResult = 100;
const MIN_CUDA_DRIVER_VERSION: i32 = 12_000;
const SMOKE_WORD: u32 = 0x4e45_5256;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct NervaCudaSmokeResult {
    status: i32,
    value: u32,
}

unsafe extern "C" {
    fn nerva_cuda_smoke(out: *mut NervaCudaSmokeResult) -> c_int;
}

const SMOKE_PTX: &str = r#"
.version 6.4
.target sm_50
.address_size 64

.visible .entry nerva_smoke_kernel(
    .param .u64 out_ptr
)
{
    .reg .u64 %rd<2>;
    .reg .u32 %r<2>;
    ld.param.u64 %rd1, [out_ptr];
    mov.u32 %r1, 0x4e455256;
    st.global.u32 [%rd1], %r1;
    ret;
}
"#;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SmokeStatus {
    Ok,
    Unavailable,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaSmokeSummary {
    pub status: SmokeStatus,
    pub gpu_name: Option<String>,
    pub driver_version: Option<i32>,
    pub runtime_version: Option<i32>,
    pub compute_capability_major: Option<i32>,
    pub compute_capability_minor: Option<i32>,
    pub device_total_memory_bytes: Option<usize>,
    pub pci_bus_id: Option<String>,
    pub device_arena_bytes: usize,
    pub pinned_host_bytes: usize,
    pub kernel_value: Option<u32>,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAbiSmokeSummary {
    pub return_code: i32,
    pub status: i32,
    pub value: u32,
    pub matched: bool,
}

impl NativeAbiSmokeSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"return_code\":{},\"status\":{},\"value\":{},\"matched\":{}}}",
            self.return_code, self.status, self.value, self.matched,
        )
    }
}

pub fn native_abi_smoke() -> NativeAbiSmokeSummary {
    let mut out = NervaCudaSmokeResult {
        status: -1,
        value: 0,
    };
    let return_code = unsafe { nerva_cuda_smoke(&mut out) };
    NativeAbiSmokeSummary {
        return_code,
        status: out.status,
        value: out.value,
        matched: return_code == 0 && out.status == 0 && out.value == SMOKE_WORD,
    }
}

impl CudaSmokeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"gpu_name\":{},\"driver_version\":{},\"runtime_version\":{},\"compute_capability_major\":{},\"compute_capability_minor\":{},\"device_total_memory_bytes\":{},\"pci_bus_id\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"kernel_value\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            json_opt_str(self.gpu_name.as_deref()),
            json_opt_i32(self.driver_version),
            json_opt_i32(self.runtime_version),
            json_opt_i32(self.compute_capability_major),
            json_opt_i32(self.compute_capability_minor),
            json_opt_usize(self.device_total_memory_bytes),
            json_opt_str(self.pci_bus_id.as_deref()),
            self.device_arena_bytes,
            self.pinned_host_bytes,
            json_opt_u32(self.kernel_value),
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }

    fn unavailable(error: impl Into<String>) -> Self {
        Self {
            status: SmokeStatus::Unavailable,
            gpu_name: None,
            driver_version: None,
            runtime_version: cuda_runtime_version(),
            compute_capability_major: None,
            compute_capability_minor: None,
            device_total_memory_bytes: None,
            pci_bus_id: None,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            kernel_value: None,
            hot_path_allocations: 0,
            error: Some(error.into()),
        }
    }

    fn failed(error: impl Into<String>) -> Self {
        Self {
            status: SmokeStatus::Failed,
            gpu_name: None,
            driver_version: None,
            runtime_version: cuda_runtime_version(),
            compute_capability_major: None,
            compute_capability_minor: None,
            device_total_memory_bytes: None,
            pci_bus_id: None,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            kernel_value: None,
            hot_path_allocations: 0,
            error: Some(error.into()),
        }
    }
}

pub fn smoke() -> CudaSmokeSummary {
    match run_smoke() {
        Ok(summary) => summary,
        Err(SmokeError::Unavailable(reason)) => CudaSmokeSummary::unavailable(reason),
        Err(SmokeError::Failed(reason)) => CudaSmokeSummary::failed(reason),
    }
}

#[derive(Debug)]
enum SmokeError {
    Unavailable(String),
    Failed(String),
}

impl fmt::Display for SmokeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmokeError::Unavailable(reason) | SmokeError::Failed(reason) => f.write_str(reason),
        }
    }
}

struct DlLibrary {
    handle: platform::LibraryHandle,
}

impl DlLibrary {
    fn open(names: &[&CStr]) -> Result<Self, SmokeError> {
        for name in names {
            if let Some(handle) = platform::open(name) {
                return Ok(Self { handle });
            }
        }
        let candidates = names
            .iter()
            .map(|name| name.to_string_lossy())
            .collect::<Vec<_>>()
            .join(", ");
        let detail = platform::last_error()
            .map(|error| format!("; last loader error: {error}"))
            .unwrap_or_default();
        Err(SmokeError::Unavailable(format!(
            "could not load CUDA shared library from [{candidates}]{detail}"
        )))
    }

    fn symbol<T: Copy>(&self, name: &'static CStr) -> Result<T, SmokeError> {
        self.symbol_any(&[name])
    }

    fn symbol_any<T: Copy>(&self, names: &[&'static CStr]) -> Result<T, SmokeError> {
        for name in names {
            if let Some(symbol) = platform::symbol(self.handle, name) {
                let typed = unsafe { std::mem::transmute_copy::<*mut c_void, T>(&symbol) };
                return Ok(typed);
            }
        }
        let candidates = names
            .iter()
            .map(|name| name.to_string_lossy())
            .collect::<Vec<_>>()
            .join(", ");
        Err(SmokeError::Failed(format!(
            "missing CUDA symbol [{candidates}]"
        )))
    }

    fn symbol_opt<T: Copy>(&self, name: &'static CStr) -> Option<T> {
        platform::symbol(self.handle, name)
            .map(|symbol| unsafe { std::mem::transmute_copy::<*mut c_void, T>(&symbol) })
    }
}

impl Drop for DlLibrary {
    fn drop(&mut self) {
        platform::close(self.handle);
    }
}

struct CudaDriver {
    _lib: DlLibrary,
    cu_get_error_name: Option<unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult>,
    cu_init: unsafe extern "C" fn(c_uint) -> CuResult,
    cu_driver_get_version: unsafe extern "C" fn(*mut c_int) -> CuResult,
    cu_device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    cu_device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    cu_device_total_mem: unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult,
    cu_device_get_pci_bus_id:
        Option<unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult>,
    cu_device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    cu_device_primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
    cu_device_primary_ctx_release: unsafe extern "C" fn(CuDevice) -> CuResult,
    cu_ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    cu_ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    cu_mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    cu_mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    cu_mem_alloc_host: unsafe extern "C" fn(*mut *mut c_void, usize) -> CuResult,
    cu_mem_free_host: unsafe extern "C" fn(*mut c_void) -> CuResult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    cu_module_load_data: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    cu_module_get_function:
        unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    cu_module_unload: unsafe extern "C" fn(CuModule) -> CuResult,
    cu_launch_kernel: unsafe extern "C" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CuStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
}

impl CudaDriver {
    fn load() -> Result<Self, SmokeError> {
        let lib = DlLibrary::open(cuda_driver_library_names())?;
        Ok(Self {
            cu_get_error_name: lib.symbol_opt(c"cuGetErrorName"),
            cu_init: lib.symbol(c"cuInit")?,
            cu_driver_get_version: lib.symbol(c"cuDriverGetVersion")?,
            cu_device_get: lib.symbol(c"cuDeviceGet")?,
            cu_device_get_attribute: lib.symbol(c"cuDeviceGetAttribute")?,
            cu_device_total_mem: lib.symbol_any(&[c"cuDeviceTotalMem_v2", c"cuDeviceTotalMem"])?,
            cu_device_get_pci_bus_id: lib.symbol_opt(c"cuDeviceGetPCIBusId"),
            cu_device_get_name: lib.symbol(c"cuDeviceGetName")?,
            cu_device_primary_ctx_retain: lib.symbol(c"cuDevicePrimaryCtxRetain")?,
            cu_device_primary_ctx_release: lib.symbol_any(&[
                c"cuDevicePrimaryCtxRelease_v2",
                c"cuDevicePrimaryCtxRelease",
            ])?,
            cu_ctx_set_current: lib.symbol(c"cuCtxSetCurrent")?,
            cu_ctx_synchronize: lib.symbol(c"cuCtxSynchronize")?,
            cu_mem_alloc: lib.symbol_any(&[c"cuMemAlloc_v2", c"cuMemAlloc"])?,
            cu_mem_free: lib.symbol_any(&[c"cuMemFree_v2", c"cuMemFree"])?,
            cu_mem_alloc_host: lib.symbol_any(&[c"cuMemAllocHost_v2", c"cuMemAllocHost"])?,
            cu_mem_free_host: lib.symbol(c"cuMemFreeHost")?,
            cu_memcpy_dtoh: lib.symbol_any(&[c"cuMemcpyDtoH_v2", c"cuMemcpyDtoH"])?,
            cu_module_load_data: lib.symbol(c"cuModuleLoadData")?,
            cu_module_get_function: lib.symbol(c"cuModuleGetFunction")?,
            cu_module_unload: lib.symbol(c"cuModuleUnload")?,
            cu_launch_kernel: lib.symbol(c"cuLaunchKernel")?,
            _lib: lib,
        })
    }

    fn check(&self, result: CuResult, op: &'static str) -> Result<(), SmokeError> {
        if result == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(SmokeError::Failed(format!(
                "{op} failed with CUDA result {}",
                self.describe_result(result)
            )))
        }
    }

    fn describe_result(&self, result: CuResult) -> String {
        let mut name = ptr::null();
        if let Some(cu_get_error_name) = self.cu_get_error_name {
            let name_result = unsafe { cu_get_error_name(result, &mut name) };
            if name_result == CUDA_SUCCESS && !name.is_null() {
                let label = unsafe { CStr::from_ptr(name) }.to_string_lossy();
                return format!("{label} ({result})");
            }
        }
        result.to_string()
    }
}

struct DeviceAllocation<'a> {
    driver: &'a CudaDriver,
    ptr: CuDevicePtr,
}

impl<'a> DeviceAllocation<'a> {
    fn new(driver: &'a CudaDriver, bytes: usize) -> Result<Self, SmokeError> {
        let mut ptr = 0;
        let result = unsafe { (driver.cu_mem_alloc)(&mut ptr, bytes) };
        driver.check(result, "cuMemAlloc")?;
        Ok(Self { driver, ptr })
    }
}

impl Drop for DeviceAllocation<'_> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                let _ = (self.driver.cu_mem_free)(self.ptr);
            }
        }
    }
}

struct PinnedAllocation<'a> {
    driver: &'a CudaDriver,
    ptr: *mut c_void,
}

impl<'a> PinnedAllocation<'a> {
    fn new(driver: &'a CudaDriver, bytes: usize) -> Result<Self, SmokeError> {
        let mut ptr = ptr::null_mut();
        let result = unsafe { (driver.cu_mem_alloc_host)(&mut ptr, bytes) };
        driver.check(result, "cuMemAllocHost")?;
        Ok(Self { driver, ptr })
    }
}

impl Drop for PinnedAllocation<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = (self.driver.cu_mem_free_host)(self.ptr);
            }
        }
    }
}

struct LoadedModule<'a> {
    driver: &'a CudaDriver,
    module: CuModule,
}

impl<'a> LoadedModule<'a> {
    fn new(driver: &'a CudaDriver, ptx: &str) -> Result<Self, SmokeError> {
        let c_ptx = CString::new(ptx).map_err(|_| {
            SmokeError::Failed("embedded smoke PTX contained a nul byte".to_string())
        })?;
        let mut module = ptr::null_mut();
        let result = unsafe { (driver.cu_module_load_data)(&mut module, c_ptx.as_ptr().cast()) };
        driver.check(result, "cuModuleLoadData")?;
        Ok(Self { driver, module })
    }

    fn function(&self, name: &CStr) -> Result<CuFunction, SmokeError> {
        let mut function = ptr::null_mut();
        let result = unsafe {
            (self.driver.cu_module_get_function)(&mut function, self.module, name.as_ptr())
        };
        self.driver.check(result, "cuModuleGetFunction")?;
        Ok(function)
    }
}

impl Drop for LoadedModule<'_> {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                let _ = (self.driver.cu_module_unload)(self.module);
            }
        }
    }
}

fn run_smoke() -> Result<CudaSmokeSummary, SmokeError> {
    let driver = CudaDriver::load()?;
    let result = unsafe { (driver.cu_init)(0) };
    if result == CUDA_ERROR_NO_DEVICE {
        return Err(SmokeError::Unavailable(format!(
            "cuInit reported {}; no CUDA compute device is accessible to this process. Check /dev/nvidia* visibility and CUDA_VISIBLE_DEVICES.",
            driver.describe_result(result)
        )));
    }
    driver.check(result, "cuInit")?;

    let mut device = 0;
    let result = unsafe { (driver.cu_device_get)(&mut device, 0) };
    if result == CUDA_ERROR_INVALID_DEVICE {
        return Err(SmokeError::Unavailable(format!(
            "cuDeviceGet(0) reported {}",
            driver.describe_result(result)
        )));
    }
    driver.check(result, "cuDeviceGet")?;

    let mut driver_version = 0;
    let result = unsafe { (driver.cu_driver_get_version)(&mut driver_version) };
    driver.check(result, "cuDriverGetVersion")?;
    if driver_version < MIN_CUDA_DRIVER_VERSION {
        return Err(SmokeError::Failed(format!(
            "NERVA CUDA supports CUDA 12.x/13.x only; driver API version is {driver_version}"
        )));
    }

    let (compute_capability_major, compute_capability_minor) = compute_capability(&driver, device)?;
    let device_total_memory_bytes = device_total_memory(&driver, device)?;
    let pci_bus_id = pci_bus_id(&driver, device);

    let mut name_buf = [0 as c_char; 128];
    let result = unsafe { (driver.cu_device_get_name)(name_buf.as_mut_ptr(), 128, device) };
    driver.check(result, "cuDeviceGetName")?;
    let gpu_name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    let mut ctx = ptr::null_mut();
    // Match rvLLM's CUDA 13-safe bring-up: retain the primary context
    // and make it current, instead of creating a legacy private context.
    let result = unsafe { (driver.cu_device_primary_ctx_retain)(&mut ctx, device) };
    driver.check(result, "cuDevicePrimaryCtxRetain")?;

    let smoke_result = (|| {
        let result = unsafe { (driver.cu_ctx_set_current)(ctx) };
        driver.check(result, "cuCtxSetCurrent")?;

        let device_word = DeviceAllocation::new(&driver, 4)?;
        let host_word = PinnedAllocation::new(&driver, 4)?;
        let module = LoadedModule::new(&driver, SMOKE_PTX)?;
        let function = module.function(c"nerva_smoke_kernel")?;

        let mut out_arg = device_word.ptr;
        let mut args = [(&mut out_arg as *mut CuDevicePtr).cast::<c_void>()];
        let result = unsafe {
            (driver.cu_launch_kernel)(
                function,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                ptr::null_mut(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        driver.check(result, "cuLaunchKernel")?;
        let result = unsafe { (driver.cu_ctx_synchronize)() };
        driver.check(result, "cuCtxSynchronize")?;

        let result = unsafe { (driver.cu_memcpy_dtoh)(host_word.ptr, device_word.ptr, 4) };
        driver.check(result, "cuMemcpyDtoH")?;
        let value = unsafe { *(host_word.ptr.cast::<u32>()) };
        if value != SMOKE_WORD {
            return Err(SmokeError::Failed(format!(
                "smoke kernel wrote 0x{value:08x}, expected 0x{SMOKE_WORD:08x}"
            )));
        }

        Ok(CudaSmokeSummary {
            status: SmokeStatus::Ok,
            gpu_name: Some(gpu_name),
            driver_version: Some(driver_version),
            runtime_version: cuda_runtime_version(),
            compute_capability_major: Some(compute_capability_major),
            compute_capability_minor: Some(compute_capability_minor),
            device_total_memory_bytes: Some(device_total_memory_bytes),
            pci_bus_id,
            device_arena_bytes: 4,
            pinned_host_bytes: 4,
            kernel_value: Some(value),
            hot_path_allocations: 0,
            error: None,
        })
    })();

    let _ = unsafe { (driver.cu_device_primary_ctx_release)(device) };
    smoke_result
}

fn compute_capability(driver: &CudaDriver, device: CuDevice) -> Result<(i32, i32), SmokeError> {
    let mut major = 0;
    let result = unsafe {
        (driver.cu_device_get_attribute)(
            &mut major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        )
    };
    driver.check(result, "cuDeviceGetAttribute(CC_MAJOR)")?;

    let mut minor = 0;
    let result = unsafe {
        (driver.cu_device_get_attribute)(
            &mut minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        )
    };
    driver.check(result, "cuDeviceGetAttribute(CC_MINOR)")?;
    Ok((major, minor))
}

fn device_total_memory(driver: &CudaDriver, device: CuDevice) -> Result<usize, SmokeError> {
    let mut bytes = 0;
    let result = unsafe { (driver.cu_device_total_mem)(&mut bytes, device) };
    driver.check(result, "cuDeviceTotalMem")?;
    Ok(bytes)
}

fn pci_bus_id(driver: &CudaDriver, device: CuDevice) -> Option<String> {
    let get_pci_bus_id = driver.cu_device_get_pci_bus_id?;
    let mut buf = [0 as c_char; 32];
    let result = unsafe { get_pci_bus_id(buf.as_mut_ptr(), buf.len() as c_int, device) };
    if result != CUDA_SUCCESS {
        return None;
    }
    let value = unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    (!value.is_empty()).then_some(value)
}

fn cuda_runtime_version() -> Option<i32> {
    let lib = DlLibrary::open(cuda_runtime_library_names()).ok()?;
    let version_fn: unsafe extern "C" fn(*mut c_int) -> c_int =
        lib.symbol(c"cudaRuntimeGetVersion").ok()?;
    let mut version = 0;
    let result = unsafe { version_fn(&mut version) };
    (result == 0).then_some(version)
}

fn cuda_driver_library_names() -> &'static [&'static CStr] {
    #[cfg(target_os = "windows")]
    {
        &[c"nvcuda.dll"]
    }
    #[cfg(target_os = "macos")]
    {
        &[
            c"/usr/local/cuda/lib/libcuda.dylib",
            c"/usr/local/cuda/lib/libcuda.1.dylib",
            c"libcuda.dylib",
        ]
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        &[
            c"/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            c"/lib/x86_64-linux-gnu/libcuda.so.1",
            c"/usr/lib/aarch64-linux-gnu/libcuda.so.1",
            c"/lib/aarch64-linux-gnu/libcuda.so.1",
            c"/usr/lib/aarch64-linux-gnu/tegra/libcuda.so",
            c"/usr/lib/wsl/lib/libcuda.so.1",
            c"libcuda.so.1",
            c"libcuda.so",
        ]
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        &[]
    }
}

fn cuda_runtime_library_names() -> &'static [&'static CStr] {
    #[cfg(target_os = "windows")]
    {
        &[c"cudart64_130.dll", c"cudart64_120.dll"]
    }
    #[cfg(target_os = "macos")]
    {
        &[c"/usr/local/cuda/lib/libcudart.dylib", c"libcudart.dylib"]
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        &[c"libcudart.so.13", c"libcudart.so.12", c"libcudart.so"]
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        &[]
    }
}

fn json_opt_i32(value: Option<i32>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_opt_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_opt_str(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", escape_json(value)),
        None => "null".to_string(),
    }
}

fn escape_json(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(unix)]
mod platform {
    use super::*;

    const RTLD_NOW: c_int = 2;

    #[cfg_attr(any(target_os = "linux", target_os = "android"), link(name = "dl"))]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *const c_char;
    }

    #[derive(Copy, Clone)]
    pub struct LibraryHandle(*mut c_void);

    pub fn open(name: &CStr) -> Option<LibraryHandle> {
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
        (!handle.is_null()).then_some(LibraryHandle(handle))
    }

    pub fn symbol(handle: LibraryHandle, name: &CStr) -> Option<*mut c_void> {
        let symbol = unsafe { dlsym(handle.0, name.as_ptr()) };
        (!symbol.is_null()).then_some(symbol)
    }

    pub fn close(handle: LibraryHandle) {
        if !handle.0.is_null() {
            unsafe {
                let _ = dlclose(handle.0);
            }
        }
    }

    pub fn last_error() -> Option<String> {
        let err = unsafe { dlerror() };
        if err.is_null() {
            None
        } else {
            Some(
                unsafe { CStr::from_ptr(err) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use super::*;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(file_name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, proc_name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> c_int;
        fn GetLastError() -> u32;
    }

    #[derive(Copy, Clone)]
    pub struct LibraryHandle(*mut c_void);

    pub fn open(name: &CStr) -> Option<LibraryHandle> {
        let handle = unsafe { LoadLibraryA(name.as_ptr()) };
        (!handle.is_null()).then_some(LibraryHandle(handle))
    }

    pub fn symbol(handle: LibraryHandle, name: &CStr) -> Option<*mut c_void> {
        let symbol = unsafe { GetProcAddress(handle.0, name.as_ptr()) };
        (!symbol.is_null()).then_some(symbol)
    }

    pub fn close(handle: LibraryHandle) {
        if !handle.0.is_null() {
            unsafe {
                let _ = FreeLibrary(handle.0);
            }
        }
    }

    pub fn last_error() -> Option<String> {
        let code = unsafe { GetLastError() };
        (code != 0).then(|| format!("GetLastError={code}"))
    }
}

#[cfg(not(any(unix, target_os = "windows")))]
mod platform {
    use super::*;

    #[derive(Copy, Clone)]
    pub struct LibraryHandle;

    pub fn open(_name: &CStr) -> Option<LibraryHandle> {
        None
    }

    pub fn symbol(_handle: LibraryHandle, _name: &CStr) -> Option<*mut c_void> {
        None
    }

    pub fn close(_handle: LibraryHandle) {}

    pub fn last_error() -> Option<String> {
        Some(format!(
            "dynamic CUDA loading is not implemented for {}",
            std::env::consts::OS
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escapes_control_chars() {
        assert_eq!(escape_json("a\"b\\c\n"), "a\\\"b\\\\c\\n");
    }

    #[test]
    fn unavailable_summary_is_valid_shape() {
        let summary = CudaSmokeSummary::unavailable("no cuda");
        let json = summary.to_json();
        assert!(json.contains("\"status\":\"unavailable\""));
        assert!(json.contains("\"compute_capability_major\":null"));
        assert!(json.contains("\"compute_capability_minor\":null"));
        assert!(json.contains("\"device_total_memory_bytes\":null"));
        assert!(json.contains("\"pci_bus_id\":null"));
        assert!(json.contains("\"hot_path_allocations\":0"));
    }

    #[test]
    fn known_cuda_platforms_have_driver_candidates() {
        if cfg!(any(unix, target_os = "windows")) {
            assert!(!cuda_driver_library_names().is_empty());
        }
    }

    #[test]
    fn native_abi_smoke_returns_expected_word() {
        let summary = native_abi_smoke();
        assert_eq!(summary.return_code, 0);
        assert_eq!(summary.status, 0);
        assert_eq!(summary.value, SMOKE_WORD);
        assert!(summary.matched);
        assert!(summary.to_json().contains("\"matched\":true"));
    }
}
