#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

// Include external HIP kernels from src directory  
#include "src/MemoryLevelSpeedTest.hip"

#define CHECK(cmd) { \
    hipError_t err = (cmd); \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define SIZE_T(x) ((size_t)(x))

// Kernel for latency measurement
__global__ void latency_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        volatile float x = data[idx];
        data[idx] = x + 1.0f;
    }
}

// Kernel for register latency measurement
__global__ void register_latency_kernel(float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        volatile float x = 1.0f;
        for (int i = 0; i < 1000; i++) {
            x = x * 1.001f + 0.001f;
        }
        output[idx] = x;
    }
}

void run_benchmark(size_t size_BYTES, int N) {
    size_t size = size_BYTES;
    
    std::cout << "\n========== Data Size: " << (size / (1024.0 * 1024.0)) << " MB ==========" << std::endl;

    // Allocate CPU RAM (Host)
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // Allocate GPU RAM (Global Device)
    float *d_A, *d_B, *d_C;
    CHECK(hipMalloc(&d_A, size));
    CHECK(hipMalloc(&d_B, size));
    CHECK(hipMalloc(&d_C, size));

    hipEvent_t start, stop;
    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&stop));
    float milliseconds = 0;

    // Test 1: PCIe Bandwidth AND Latency (Host to Device)
    CHECK(hipEventRecord(start));
    CHECK(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));
    CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    double pcie_bw = (size / (1024.0 * 1024.0 * 1024.0)) / (milliseconds / 1000.0);
    double PCIe_latency_sec = milliseconds / 1000.0;
    std::cout << "\n[PCIe] CPU RAM <-> GPU RAM:" << std::endl;
    std::cout << "   Bandwidth: " << std::fixed << std::setprecision(2) << pcie_bw << " GB/s" << std::endl;
    std::cout << "   Latency:   " << std::fixed << std::setprecision(3) << (milliseconds * 1000.0) << " microseconds" << std::endl;

    // Test 2: Global Memory Bandwidth AND Latency
    CHECK(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CHECK(hipEventRecord(start));
    global_mem_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));
    CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    double global_bw = (3.0 * size / (1024.0 * 1024.0 * 1024.0)) / (milliseconds / 1000.0);
    std::cout << "\n[GPU Global Memory] VRAM:" << std::endl;
    std::cout << "   Bandwidth: " << std::fixed << std::setprecision(2) << global_bw << " GB/s" << std::endl;
    std::cout << "   Kernel Latency: " << std::fixed << std::setprecision(3) << (milliseconds * 1000.0) << " microseconds" << std::endl;

    // Test 3: Shared Memory (LDS) Operations - measure time only
    CHECK(hipEventRecord(start));
    shared_mem_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));
    CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    std::cout << "\n[GPU Shared Memory] LDS:" << std::endl;
    std::cout << "   Kernel Time: " << std::fixed << std::setprecision(3) << milliseconds << " ms" << std::endl;

    // Test 4: Register / ALU Speed - measure latency
    CHECK(hipEventRecord(start));
    register_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, N);
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));
    CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    double gflops = (N * 10000.0 * 4.0) / (milliseconds / 1000.0) / 1e9;
    std::cout << "\n[GPU Register] L1/Register:" << std::endl;
    std::cout << "   Math Throughput: " << std::fixed << std::setprecision(2) << gflops << " GFLOP/s" << std::endl;
    std::cout << "   Kernel Latency:  " << std::fixed << std::setprecision(3) << (milliseconds * 1000.0) << " microseconds" << std::endl;

    // Test 5: CPU RAM Latency (memory access latency)
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        volatile float x = h_A[i] + h_B[i];
        h_C[i] = x;
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    double cpu_latency_per_element = (cpu_time_ms * 1000.0) / N;
    std::cout << "\n[CPU RAM] Host Memory:" << std::endl;
    std::cout << "   Total Access Time: " << std::fixed << std::setprecision(3) << cpu_time_ms << " ms" << std::endl;
    std::cout << "   Avg Latency/Elem:  " << std::fixed << std::setprecision(6) << cpu_latency_per_element << " microseconds" << std::endl;

    // Test 6: Fine-grained latency measurement with small transfers
    size_t small_size = sizeof(float); // Single float transfer for latency
    float *h_small = new float[1];
    float *d_small;
    CHECK(hipMalloc(&d_small, small_size));
    
    int iterations = 100;
    float total_latency = 0;
    for (int i = 0; i < iterations; i++) {
        h_small[0] = static_cast<float>(i);
        CHECK(hipEventRecord(start));
        CHECK(hipMemcpy(d_small, h_small, small_size, hipMemcpyHostToDevice));
        CHECK(hipEventRecord(stop));
        CHECK(hipEventSynchronize(stop));
        CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        total_latency += milliseconds;
    }
    double avg_transfer_latency = (total_latency / iterations) * 1000.0; // microseconds
    std::cout << "\n[Fine-grained Latency]" << std::endl;
    std::cout << "   PCIe Single Transfer: " << std::fixed << std::setprecision(3) << avg_transfer_latency << " microseconds" << std::endl;
    
    // GPU kernel launch latency
    CHECK(hipEventRecord(start));
    latency_kernel<<<1, 1>>>(d_small, 1);
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));
    CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    std::cout << "   GPU Kernel Launch:    " << std::fixed << std::setprecision(3) << (milliseconds * 1000.0) << " microseconds" << std::endl;

    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C;
    CHECK(hipFree(d_A)); CHECK(hipFree(d_B)); CHECK(hipFree(d_C));
    CHECK(hipEventDestroy(start)); CHECK(hipEventDestroy(stop));
    delete[] h_small;
    CHECK(hipFree(d_small));
}

int main() {
    std::cout << "=== Memory Hierarchy Latency & Bandwidth Benchmark ===" << std::endl;
    std::cout << "Testing different data sizes..." << std::endl;

    // Three test sizes: 64MB, 1GB, 10GB
    struct {
        size_t size_BYTES;
        size_t N;
        std::string name;
    } test_cases[] = {
        {SIZE_T(64) * 1024 * 1024, SIZE_T(64) * 1024 * 1024 / sizeof(float), "64 MB"},
        {SIZE_T(1024) * 1024 * 1024, SIZE_T(1024) * 1024 * 1024 / sizeof(float), "1 GB"},
        {SIZE_T(10) * 1024 * 1024 * 1024, SIZE_T(10) * 1024 * 1024 * 1024 / sizeof(float), "10 GB"}
    };

#define SIZE_T(x) ((size_t)(x))

    // Check GPU memory before running large tests
    size_t free_mem, total_mem;
    CHECK(hipMemGetInfo(&free_mem, &total_mem));
    std::cout << "\nGPU Memory: " << (free_mem / (1024.0 * 1024.0 * 1024.0)) << " GB free / " 
              << (total_mem / (1024.0 * 1024.0 * 1024.0)) << " GB total" << std::endl;

    for (auto &test : test_cases) {
        // Check if we have enough memory
        size_t required = test.size_BYTES * 3; // 3 arrays
        if (required > free_mem * 0.9) {
            std::cout << "\n[SKIPPED] " << test.name << " - Not enough GPU memory" << std::endl;
            continue;
        }
        run_benchmark(test.size_BYTES, test.N);
    }

    return 0;
}
