#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace face_pipeline::utils {

/// Check a CUDA call and throw std::runtime_error on failure.
inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                                 std::to_string(line) + " - " + cudaGetErrorString(err));
    }
}

/// RAII wrapper for a CUDA stream.
class CudaStream {
public:
    CudaStream() { cuda_check(cudaStreamCreate(&stream_), __FILE__, __LINE__); }
    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const { cuda_check(cudaStreamSynchronize(stream_), __FILE__, __LINE__); }

private:
    cudaStream_t stream_ = nullptr;
};

/// RAII wrapper for pinned (page-locked) host memory.
template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t count) : count_(count) {
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                   __FILE__, __LINE__);
    }
    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFreeHost(data_);
            }
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace face_pipeline::utils

#define FP_CUDA_CHECK(call) ::face_pipeline::utils::cuda_check((call), __FILE__, __LINE__)
