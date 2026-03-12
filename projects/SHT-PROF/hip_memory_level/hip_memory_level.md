Here is a complete lesson plan and structure for your Jupyter Notebook, written in English as requested. It includes detailed explanations of the memory hierarchy, conceptual introductions, and a standalone HIP C++ code block designed to measure bandwidth and relative access speeds.

You can copy and paste the markdown sections below directly into your Jupyter Notebook cells.

---

# Notebook Title: Exploring AMD Radeon GPU Memory Hierarchy with HIP

### Cell 1: Markdown (Introduction & Architecture)

## 1. Introduction to the AMD GPU Memory Hierarchy

To write highly optimized code for AMD Radeon GPUs (or any modern GPU), understanding the memory hierarchy is essential. A GPU is a massive throughput engine, but its arithmetic logic units (ALUs) can only perform calculations as fast as data is fed to them.

The memory hierarchy is designed as a pyramid: the closer the memory is to the compute cores, the faster it is (lower latency, higher bandwidth), but the smaller its capacity. In HIP (which maps directly to AMD's hardware), we interact with four primary levels of memory:

1. **CPU RAM (Host Memory):**
* **Location:** Off-GPU, physically located on the motherboard.
* **Connection:** Connected to the GPU via the PCIe bus.
* **Characteristics:** Largest capacity (System RAM), but incredibly slow to access from the GPU. Transferring data across the PCIe bus has the highest latency and lowest bandwidth.


2. **GPU Global Memory (Device Memory / VRAM):**
* **Location:** On the GPU board (e.g., GDDR6 or HBM).
* **Characteristics:** Large capacity (typically 8GB to 192GB). Accessible by all threads across all Compute Units (CUs). It has high bandwidth (hundreds of GB/s to TB/s) but high latency (hundreds of clock cycles).


3. **GPU Shared Memory (Local Data Share - LDS):**
* **Location:** On-chip memory within a Compute Unit (CU).
* **Characteristics:** Small capacity (usually 64KB per CU). It is shared only among threads within the same Workgroup (Block). Because it is on-chip, it has extremely low latency and massive bandwidth. It is user-managed cache.


4. **GPU Registers (Vector General-Purpose Registers - VGPRs):**
* **Location:** Directly attached to the ALUs inside the SIMD units.
* **Characteristics:** The fastest memory available. Dedicated entirely to a single thread (private memory). Accessing a register generally takes 0 extra clock cycles (latency is hidden by the pipeline). However, using too many registers limits the number of threads that can run concurrently (occupancy).



---

### Cell 2: Markdown (Concepts)

## 2. Bandwidth vs. Latency

When measuring memory performance, we look at two metrics:

* **Bandwidth (Throughput):** How much data can be moved per second (measured in GB/s). Imagine this as the width of a highway.
* **Latency:** The time it takes for a single data request to be fulfilled (measured in nanoseconds or clock cycles). Imagine this as the speed limit on that highway.

In the following HIP code, we will measure the **Bandwidth** of Host-to-Device transfers, Global Memory, and Shared Memory. We will also proxy **Register latency/speed** by measuring the compute throughput (GFLOPs) when data is held entirely in registers.

---

### Cell 3: Code (The Standalone HIP Benchmark)

Save the following code in a file named `memory_hierarchy.cpp`. You can execute this block in Jupyter by using the `%%writefile memory_hierarchy.cpp` magic command at the top of the cell.

```cpp
%%writefile memory_hierarchy.cpp
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(error) << " at line " << __LINE__ << std::endl;\
        exit(-1);\
    }\
}

// ---------------------------------------------------------
// Kernel 1: Global Memory Bandwidth (Vector Addition)
// ---------------------------------------------------------
__global__ void global_mem_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Reads from global memory (A and B), writes to global memory (C)
        C[idx] = A[idx] + B[idx];
    }
}

// ---------------------------------------------------------
// Kernel 2: Shared Memory (LDS) Bandwidth
// ---------------------------------------------------------
__global__ void shared_mem_kernel(const float* A, float* C, int N) {
    // Allocate Local Data Share (LDS)
    __shared__ float lds_mem[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Load from Global to Shared
        lds_mem[tid] = A[idx];
        __syncthreads(); // Wait for all threads in workgroup

        // Perform aggressive reads/writes on Shared Memory to test bandwidth
        float temp = lds_mem[tid];
        for (int i = 0; i < 100; i++) {
            temp = temp * 1.01f;
            lds_mem[tid] = temp;
        }
        __syncthreads();

        // Write back to Global
        C[idx] = lds_mem[tid];
    }
}

// ---------------------------------------------------------
// Kernel 3: Register Speed / Compute Throughput
// ---------------------------------------------------------
__global__ void register_kernel(float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Variables declared here reside in thread-private Registers
        float r1 = 1.0f;
        float r2 = 1.00001f;
        float r3 = 0.5f;

        // Force intensive register-to-register operations
        // FMA (Fused Multiply-Add) maps nicely to hardware
        for(int i = 0; i < 10000; i++) {
            r1 = r1 * r2 + r3;
            r2 = r2 * 0.99999f + r1;
        }

        C[idx] = r1 + r2; // Write out to prevent compiler from optimizing the loop away
    }
}

// ---------------------------------------------------------
// Main Execution Host Code
// ---------------------------------------------------------
int main() {
    int N = 1 << 24; // ~16 Million elements
    size_t size = N * sizeof(float);

    std::cout << "--- AMD Radeon Memory Hierarchy Benchmark ---" << std::endl;
    std::cout << "Data size per array: " << size / (1024.0 * 1024.0) << " MB" << std::endl;

    // 1. Allocate CPU RAM (Host)
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // 2. Allocate GPU RAM (Global Device)
    float *d_A, *d_B, *d_C;
    CHECK(hipMalloc(&d_A, size));
    CHECK(hipMalloc(&d_B, size));
    CHECK(hipMalloc(&d_C, size));

    hipEvent_t start, stop;
    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&stop));
    float milliseconds = 0;

    // --- TEST 1: PCIe Bandwidth (Host to Device) ---
    CHECK(hipEventRecord(start));
    CHECK(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));
    CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    double pcie_bw = (size / (1024.0 * 1024.0 * 1024.0)) / (milliseconds / 1000.0);
    std::cout << "\n[1] CPU RAM -> GPU RAM (PCIe) Bandwidth:  " << pcie_bw << " GB/s" << std::endl;

    // --- TEST 2: Global Memory Bandwidth ---
    CHECK(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CHECK(hipEventRecord(start));
    global_mem_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));
    CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    // Data moved: 2 reads (A, B) + 1 write (C)
    double global_bw = (3.0 * size / (1024.0 * 1024.0 * 1024.0)) / (milliseconds / 1000.0);
    std::cout << "[2] GPU Global Memory (VRAM) Bandwidth: " << global_bw << " GB/s" << std::endl;

    // --- TEST 3: Shared Memory (LDS) Operations ---
    CHECK(hipEventRecord(start));
    shared_mem_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));
    CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    std::cout << "[3] GPU Shared Memory Kernel Time:      " << milliseconds << " ms (Noticeably fast given the loop count)" << std::endl;

    // --- TEST 4: Register / ALU Speed ---
    CHECK(hipEventRecord(start));
    register_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, N);
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));
    CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    // Calculate rough GFLOPs: N threads * 10000 loops * 4 FLOPs per loop
    double gflops = (N * 10000.0 * 4.0) / (milliseconds / 1000.0) / 1e9;
    std::cout << "[4] GPU Register Math Throughput:       " << gflops << " GFLOP/s" << std::endl;

    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C;
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    hipEventDestroy(start); hipEventDestroy(stop);

    return 0;
}

```

---

### Cell 4: Code (Compilation and Execution)

Use the following bash commands inside Jupyter to compile and run the benchmark. You must have the ROCm toolkit installed.

```bash
!hipcc memory_hierarchy.cpp -o memory_benchmark -O3
!./memory_benchmark

```

---

### Cell 5: Markdown (Student Exercises & Analysis)

## 3. Analysis and Questions for Students

Once you run the code above, look at the console output and reflect on the following questions:

1. **The PCIe Bottleneck:** Look at the CPU to GPU bandwidth. It is likely between 10 to 30 GB/s depending on your PCIe generation (Gen3/Gen4/Gen5). How does this compare to the Global Memory bandwidth? *Takeaway: Avoid copying data between the CPU and GPU frequently. Keep data on the GPU as long as possible.*
2. **Global Memory Constraints:** You should see Global Memory bandwidth in the hundreds of GB/s (or over 1 TB/s on high-end MI-series GPUs). While fast, the memory bus can still easily become saturated if every mathematical operation requires a trip to VRAM.
3. **The Power of Shared Memory & Registers:** In Kernels 2 and 3, we execute thousands of iterations of math loops per thread. If those loops had to read from Global Memory, the kernel would take magnitudes longer. Because the variables are kept in Registers and LDS (which sit millimeters away from the ALUs inside the chip), the operations complete almost instantaneously.

---

Would you like me to prepare a supplemental diagram illustrating how a single Thread Block (Workgroup) maps to the Compute Unit, LDS, and Registers to include in the notebook?
