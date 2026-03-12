// Global Memory Bandwidth Kernel (Vector Addition)
__global__ void global_mem_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Shared Memory (LDS) Bandwidth Kernel
__global__ void shared_mem_kernel(const float* A, float* C, int N) {
    __shared__ float lds_mem[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        lds_mem[tid] = A[idx];
        __syncthreads();
        
        float temp = lds_mem[tid];
        for (int i = 0; i < 100; i++) {
            temp = temp * 1.01f;
            lds_mem[tid] = temp;
        }
        __syncthreads();
        
        C[idx] = lds_mem[tid];
    }
}

// Register Speed / Compute Throughput Kernel
__global__ void register_kernel(float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float r1 = 1.0f;
        float r2 = 1.00001f;
        float r3 = 0.5f;
        
        for(int i = 0; i < 10000; i++) {
            r1 = r1 * r2 + r3;
            r2 = r2 * 0.99999f + r1;
        }
        
        C[idx] = r1 + r2;
    }
}
