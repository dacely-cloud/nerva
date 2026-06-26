#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

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
const RTLD_NOW: c_int = 2;
const SMOKE_WORD: u32 = 0x4e45_5256;

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

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

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
    pub device_arena_bytes: usize,
    pub pinned_host_bytes: usize,
    pub kernel_value: Option<u32>,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaSmokeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"gpu_name\":{},\"driver_version\":{},\"runtime_version\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"kernel_value\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            json_opt_str(self.gpu_name.as_deref()),
            json_opt_i32(self.driver_version),
            json_opt_i32(self.runtime_version),
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
    handle: *mut c_void,
}

impl DlLibrary {
    fn open(names: &[&CStr]) -> Result<Self, SmokeError> {
        for name in names {
            let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
            if !handle.is_null() {
                return Ok(Self { handle });
            }
        }
        Err(SmokeError::Unavailable(last_dl_error().unwrap_or_else(
            || "could not load CUDA shared library".to_string(),
        )))
    }

    fn symbol<T: Copy>(&self, name: &'static CStr) -> Result<T, SmokeError> {
        let symbol = unsafe { dlsym(self.handle, name.as_ptr()) };
        if symbol.is_null() {
            return Err(SmokeError::Failed(format!(
                "missing symbol {}",
                name.to_string_lossy()
            )));
        }
        let typed = unsafe { std::mem::transmute_copy::<*mut c_void, T>(&symbol) };
        Ok(typed)
    }
}

impl Drop for DlLibrary {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = dlclose(self.handle);
            }
        }
    }
}

struct CudaDriver {
    _lib: DlLibrary,
    cu_init: unsafe extern "C" fn(c_uint) -> CuResult,
    cu_driver_get_version: unsafe extern "C" fn(*mut c_int) -> CuResult,
    cu_device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    cu_device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
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
        let lib = DlLibrary::open(&[
            c"/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            c"/lib/x86_64-linux-gnu/libcuda.so.1",
            c"/usr/lib/aarch64-linux-gnu/libcuda.so.1",
            c"/lib/aarch64-linux-gnu/libcuda.so.1",
            c"libcuda.so.1",
            c"libcuda.so",
        ])?;
        Ok(Self {
            cu_init: lib.symbol(c"cuInit")?,
            cu_driver_get_version: lib.symbol(c"cuDriverGetVersion")?,
            cu_device_get_count: lib.symbol(c"cuDeviceGetCount")?,
            cu_device_get: lib.symbol(c"cuDeviceGet")?,
            cu_device_get_name: lib.symbol(c"cuDeviceGetName")?,
            cu_device_primary_ctx_retain: lib.symbol(c"cuDevicePrimaryCtxRetain")?,
            cu_device_primary_ctx_release: lib.symbol(c"cuDevicePrimaryCtxRelease_v2")?,
            cu_ctx_set_current: lib.symbol(c"cuCtxSetCurrent")?,
            cu_ctx_synchronize: lib.symbol(c"cuCtxSynchronize")?,
            cu_mem_alloc: lib.symbol(c"cuMemAlloc_v2")?,
            cu_mem_free: lib.symbol(c"cuMemFree_v2")?,
            cu_mem_alloc_host: lib.symbol(c"cuMemAllocHost_v2")?,
            cu_mem_free_host: lib.symbol(c"cuMemFreeHost")?,
            cu_memcpy_dtoh: lib.symbol(c"cuMemcpyDtoH_v2")?,
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
                "{op} failed with CUDA result {result}"
            )))
        }
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
        driver.check(result, "cuMemAlloc_v2")?;
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
        driver.check(result, "cuMemAllocHost_v2")?;
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
    if result == 100 {
        return Err(SmokeError::Unavailable(
            "cuInit reported CUDA_ERROR_NO_DEVICE (100)".to_string(),
        ));
    }
    driver.check(result, "cuInit")?;

    let mut driver_version = 0;
    let result = unsafe { (driver.cu_driver_get_version)(&mut driver_version) };
    driver.check(result, "cuDriverGetVersion")?;

    let mut count = 0;
    let result = unsafe { (driver.cu_device_get_count)(&mut count) };
    driver.check(result, "cuDeviceGetCount")?;
    if count <= 0 {
        return Err(SmokeError::Unavailable(
            "CUDA driver reported zero devices".to_string(),
        ));
    }

    let mut device = 0;
    let result = unsafe { (driver.cu_device_get)(&mut device, 0) };
    driver.check(result, "cuDeviceGet")?;

    let mut name_buf = [0 as c_char; 128];
    let result = unsafe { (driver.cu_device_get_name)(name_buf.as_mut_ptr(), 128, device) };
    driver.check(result, "cuDeviceGetName")?;
    let gpu_name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    let mut ctx = ptr::null_mut();
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
        driver.check(result, "cuMemcpyDtoH_v2")?;
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

fn cuda_runtime_version() -> Option<i32> {
    let lib = DlLibrary::open(&[c"libcudart.so.12", c"libcudart.so"]).ok()?;
    let version_fn: unsafe extern "C" fn(*mut c_int) -> c_int =
        lib.symbol(c"cudaRuntimeGetVersion").ok()?;
    let mut version = 0;
    let result = unsafe { version_fn(&mut version) };
    (result == 0).then_some(version)
}

fn last_dl_error() -> Option<String> {
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

fn json_opt_i32(value: Option<i32>) -> String {
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
        assert!(json.contains("\"hot_path_allocations\":0"));
    }
}
