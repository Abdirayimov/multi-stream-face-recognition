#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward declarations to keep TensorRT out of public headers.
namespace nvinfer1 {
class ICudaEngine;
class IExecutionContext;
class IRuntime;
class ILogger;
}  // namespace nvinfer1

namespace face_pipeline::trt {

/// Description of a single I/O binding (input or output) of a TRT engine.
struct BindingInfo {
    std::string name;
    std::vector<std::int64_t> shape;  ///< Static shape; -1 indicates dynamic.
    std::size_t element_size = 0;     ///< Bytes per element (e.g. 4 for FP32).
    std::size_t volume = 0;           ///< Total element count for `shape`.
    bool is_input = false;
};

/// RAII wrapper around a TensorRT engine + execution context.
///
/// Loads a serialized `.engine` file once, owns one IExecutionContext, and
/// manages device memory for every binding. Inference is asynchronous on the
/// caller-provided CUDA stream; the caller is responsible for synchronizing.
///
/// Move-only. Not safe to share a single instance across threads concurrently;
/// use one TrtEngine per worker thread (each with its own context).
class TrtEngine {
public:
    /// Load an engine from disk.
    /// @throws std::runtime_error on file or deserialization failure.
    explicit TrtEngine(const std::string& engine_path);
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;
    TrtEngine(TrtEngine&&) noexcept;
    TrtEngine& operator=(TrtEngine&&) noexcept;

    /// Set a dynamic input shape for binding `name`. No-op for static engines.
    void set_input_shape(const std::string& name, const std::vector<std::int64_t>& shape);

    /// Copy host data into the device buffer of binding `name`.
    void copy_input(const std::string& name, const void* host_src, std::size_t bytes,
                    cudaStream_t stream);

    /// Copy device data of binding `name` back to host.
    void copy_output(const std::string& name, void* host_dst, std::size_t bytes,
                     cudaStream_t stream) const;

    /// Run inference on `stream`. Inputs must already be populated via
    /// `copy_input` (or by writing directly into `device_ptr(name)`).
    void infer(cudaStream_t stream);

    /// Direct device pointer for a binding. Useful when chaining engines that
    /// share GPU buffers without a host round-trip.
    void* device_ptr(const std::string& name);
    const void* device_ptr(const std::string& name) const;

    const std::vector<BindingInfo>& bindings() const noexcept { return bindings_; }
    const BindingInfo& binding(const std::string& name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<BindingInfo> bindings_;
};

}  // namespace face_pipeline::trt
