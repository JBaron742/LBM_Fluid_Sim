#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

// A simple CUDA kernel that adds a constant to an array
__global__ void gpu_add(int *data, int value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) { // Just have the first thread do a tiny task
        *data += value;
    }
}

int main(int argc, char** argv) {
    // 1. Initialize MPI
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 2. Setup CUDA data
    int h_val = 10;
    int *d_val;
    cudaMalloc((void**)&d_val, sizeof(int));
    cudaMemcpy(d_val, &h_val, sizeof(int), cudaMemcpyHostToDevice);

    // 3. Launch Kernel: Each rank adds its own rank ID to the value
    gpu_add<<<1, 1>>>(d_val, world_rank);
    cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Rank %d Error: %s\n", world_rank, cudaGetErrorString(err));
}

// Check for execution errors
err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
    printf("Rank %d Sync Error: %s\n", world_rank, cudaGetErrorString(err));
}
    
    // Synchronize to ensure the kernel finished
    cudaDeviceSynchronize();

    // 4. Bring the result back to CPU
    int result = 0;
    cudaMemcpy(&result, d_val, sizeof(int), cudaMemcpyDeviceToHost);

    // 5. Each rank reports its success
    printf("MPI Rank %d/%d: GPU calculated 10 + %d = %d\n", 
           world_rank, world_size, world_rank, result);

    // Cleanup
    cudaFree(d_val);
    MPI_Finalize();
    return 0;
}