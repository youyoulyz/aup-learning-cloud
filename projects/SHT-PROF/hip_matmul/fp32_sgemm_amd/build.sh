 #!/bin/bash
 export HIP_PLATFORM=amd

# export PATH=$PATH:/opt/rocm-7.2.0/bin:/opt/rocm-7.2.0/llvm/bin
export PATH=$PATH:/opt/rocm/core-7.11/bin:/opt/rocm/core-7.11/lib/llvm/bin

mkdir -p tmp
rm tmp/*.o
echo "Building Kernel0..."
hipcc -g  -c -std=c++17 -O3 --offload-arch=gfx1151 src/kernel0_rocblas.cpp -o tmp/kernel0_rocblas.o
echo "Building Kernel1..."
hipcc -g  -c -std=c++17 -O3 --offload-arch=gfx1151 src/kernel1_naive.cpp -o tmp/kernel1_naive.o
echo "Building Kernel2..."
hipcc -g  -c -std=c++17 -O3 --offload-arch=gfx1151 src/kernel2_lds.cpp -o tmp/kernel2_lds.o
echo "Building Kernel3..."
hipcc -g  -c -std=c++17 -O3 --offload-arch=gfx1151 src/kernel3_registers.cpp -o tmp/kernel_registers.o
echo "Building Kernel4..."
hipcc -g  -c -std=c++17 -O3 --offload-arch=gfx1151 src/kernel4_gmem_df.cpp -o tmp/kernel4_gmem_df.o
echo "Building Kernel5..."
hipcc -g  -c -std=c++17 -O3 --offload-arch=gfx1151 -mcumode src/kernel5_lds_optim.cpp -o tmp/kernel5_lds_optim.o
echo "Building Kernel6..."
hipcc -g  -Wunused-result -c -std=c++17 -O3 --offload-arch=gfx1151 src/kernel6_valu_optim.cpp -o tmp/kernel6_valu_optim.o
echo "Building Kernel7..."
hipcc -g  -Wunused-result -c -std=c++17 -O3 --offload-arch=gfx1151 src/kernel7_unroll.cpp -o tmp/kernel7_unroll.o
echo "Building Kernel8..."
hipcc -g  -Wunused-result -c -std=c++17 -O3 --offload-arch=gfx1151 src/kernel8_batched_gmem.cpp -o tmp/kernel8_batched_gmem.o
echo "Building host code..."
hipcc -g  -c -std=c++17 -O3 --offload-arch=gfx1151 src/main.cpp -o tmp/main.o
hipcc -g  -std=c++17 -O3 --offload-arch=gfx1151 -lrocblas tmp/*.o -o sgemm
echo "Building Kernel6 from ISA"
clang -g  -target amdgcn-amd-amdhsa -mcpu=gfx1151 -c src/kernel6_valu_optim.s -o tmp/kernel6_device_code.o
ld.lld -shared tmp/kernel6_device_code.o -o kernel6.hsaco
echo "Building Kernel7 from ISA"
clang -g  -target amdgcn-amd-amdhsa -mcpu=gfx1151 -c src/kernel7_unroll.s -o tmp/kernel7_device_code.o
ld.lld -shared tmp/kernel7_device_code.o -o kernel7.hsaco
echo "Building Kernel8 from ISA"
clang -g  -target amdgcn-amd-amdhsa -mcpu=gfx1151 -c src/kernel8_batched_gmem.s -o tmp/kernel8_device_code.o
ld.lld -shared tmp/kernel8_device_code.o -o kernel8.hsaco
echo "Build completed successfully."
