# GPU Matrix Multiplication Optimization Tutorial

## Lab Setup & Instructions

Welcome to this hands-on tutorial on GPU matrix multiplication optimization! This lab walks you through progressively optimizing a GEMM (General Matrix Multiplication) kernel on AMD GPUs, starting from a naive implementation to achieving performance that exceeds rocBLAS.

### Learning Objectives

By completing this tutorial, you will:
1. Understand the memory hierarchy on AMD GPUs (Global Memory → LDS → Registers)
2. Learn about key concepts: tiling, coalescing, bank conflicts, occupancy vs. utilization
3. Master dual-issue instructions and VGPR bank optimization
4. Achieve **50+ TFLOPS** on RDNA3 architecture through systematic optimizations

### File Structure

```
hip_matmul/
├── main.hip          # Test harness: compiles & benchmarks all kernels
├── kernel1_naive.hip    # Baseline: one thread per output element
├── kernel2_lds.hip      # Optimization: LDS tiling
├── kernel3_register.hip # Optimization: register tiling  
├── kernel4_gmem_db.hip  # Optimization: GMEM double buffering
├── kernel5_lds_opt.hip  # Optimization: bank conflict avoidance, CU mode
├── kernel6_valu.hip     # Optimization: dual FMA, VGPR banks
└── kernel7_unroll.hip   # Optimization: loop unrolling
```

### Your Task

Each `kernel*.hip` file contains sections marked with `TODO` comments. Fill in the missing code to implement each optimization step. Then run:

```bash
hipcc -O3 -mcumode -o matmul_test main.hip && ./matmul_test
```

Compare your results against rocBLAS and the expected performance targets!

---

## Background: Matrix Multiplication

Given matrices A (M×K) and B (K×N), we compute C (M×N):

```math
C[i,j] = Σ(A[i,k] × B[k,j])  for k = 0 to K-1
```

For **4096×4096** FP32 matrices:
- **Operations**: 2 × 4096³ ≈ 137 trillion FLOPs
- **Memory I/O**: 3 × 4096² × 4 bytes ≈ 200 MB

### Theoretical Peak (RDNA3 - RX 7900 XTX)

| Spec | Value |
|-- ---- |-------|
| Freq | 2.5 GHz |
| WGPs | 48 (192 SIMDs) |
| FP Ops/cycle per SIMD | 128 (dual issue FMA) |
| **Peak FLOPS** | **61.4 TFLOPS/s** |
| Memory Bandwidth | 960 GB/s |

---

## Optimization Journey Overview

| Kernel | Technique | Target Time | Key Concept |
|--------|------ ----- | ------ ---- -|-------------|
| 0 | rocBLAS (reference) | ~4.5ms | Baseline |
| 1 | Naive | ~136ms | One thread per element |
| 2 | LDS Tiling | ~34ms | Shared memory cache |
| 3 | Register Tiling | ~6ms | Multiple elements/thread |
| 4 | GMEM Double Buffer | ~5.4ms | Hide latency |
| 5 | LDS Bank Optimization | ~4.1ms | Avoid conflicts |
| 6 | VALU/Dual Issue | ~3.6ms | Dual FMA instructions |
| 7 | Loop Unrolling | ~3.3ms | More ILP |

---

## Key Concepts Explained

### Memory Hierarchy

```
┌── ────── ────── ───── ──── ── ── ───── ┐
│    Global Memory (GMEM)         │  ← 960 GB/s, high latency
│    ‣ Slow but large             │
└──────────────┬──── ───── −─────────┘
               │
        ┌──────▼── ─── ┐
        │   LDS        │  ← ~29 TB/s, low latency  
        │   ‣ Shared by workgroup
        │   ‣ 64KB per CU
        └─── ──┬─ ─────┘
               │
        ┌──────▼── ─── ┐
        │ Registers    │  ← Very fast, private per thread
        │   ‣ VGPRs: 256 max per kernel
        │   ‣ 1536 per SIMD
        └─────── ──────┘
```

### Wavefront Execution Model

- **Wave** (= CUDA warp): 32 threads executing in lock-step
- **Workgroup**: 256 threads = 8 waves
- **SIMD**: Executes one wave at a time, up to 16 waves concurrently
- **Occupancy**: #active waves / max possible waves
- **VALU Utilization**: % of time arithmetic units are busy

### Dual Issue FMA

RDNA3 can execute two independent instructions per cycle:

```assembly
v_dual_fmac_f32 v3, v4, v5 :: v_dual_fmac_f32 v6, v7, v8
```

**Constraints for dual issue**:
- Instructions must be independent
- Source registers must use different VGPR banks (bank = reg_id % 4)
- One destination even, one odd

---

## Building & Testing

### Prerequisites

```bash
# Install ROCm development tools
# Verify HIP compiler
hipcc --version

# Check GPU
rocminfo | grep "Name:"
```

### Compilation

```bash
# Basic compilation
hipcc -O3 main.hip -o matmul_test

# With CU mode (for kernel 5+)
hipcc -O3 -mcumode main.hip -o matmul_test 

# Generate ISA assembly for analysis
hipcc --save-temps -O3 -mcumode kernel5_lds_opt.hip
```

### Profiling with ROCprof

```bash
rocprof --stats-db ./run ./matmul_test
rocprof-parser stats.db
```

---

## Performance Checklist

After each kernel, check:
1. ✅ Code compiles without errors
2. ✅ Results match rocBLAS reference (use `hipDeviceSynchronize()`)
3. ✅ Performance metric is printed correctly
4. ✅ Time improves vs. previous kernel

Expected final result: **< 2.9ms** (~50 TFLOPS)

---

## Tips for Success

1. **Start simple**: Get the naive kernel working first
2. **Verify correctness**: Use small matrices (32×32) to debug
3. **Check bounds**: Ensure no out-of-bounds memory access
4. **Sync properly**: `__syncthreads()` after LDS writes
5. **Coalesced access**: Adjacent threads → adjacent memory
6. **Read RGP output**: Understand what bottleneck you're solving

---

## References

- [Original Blog](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html)
- [AMD RDNA3 ISA Guide](https://github.com/GPUOpen-Drivers/RADV/releases)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)

**Good luck and happy optimizing!** 🚀
