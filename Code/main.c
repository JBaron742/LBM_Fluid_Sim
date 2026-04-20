//dimentions of the grid
#define NX 100
#define NY 100
#define TAU 0.6f  //viscosity control (relaxation time)

//d2q9 constants

__constant__ float d_w[9];
__constant__ int d_cx[9];
__constant__ int d_cy[9];


float w[9] = {4./9, 1./9, 1./9, 1./9, 1./9, 1./36, 1./36, 1./36, 1./36};
//velocity vectors
int cx[9]  = {0, 1, 0, -1, 0, 1, -1, -1, 1};
int cy[9]  = {0, 0, 1, 0, -1, 1, 1, -1, -1};
//oppsite for each vector for bounce back
int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};


int main(int argc, char* argv[]) {
    size_t total_cells = NX * NY;
    size_t total_size = 9 * total_cells * sizeof(float);

    // Host (CPU) memory
    float *h_f = (float*)malloc(total_size);

    // Device (GPU) memory
    float *d_f, *d_f_next;
    cudaMalloc(&d_f, total_size);
    cudaMalloc(&d_f_next, total_size);

    // Initialize the grid on CPU
    initialize_grid(h_f, NX, NY);

    // Copy constants to GPU
    cudaMemcpyToSymbol(d_w, h_w, 9 * sizeof(float));
    cudaMemcpyToSymbol(d_cx, h_cx, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_cy, h_cy, 9 * sizeof(int));

    // Copy initial grid to GPU
    cudaMemcpy(d_f, h_f, total_size, cudaMemcpyHostToDevice);

    // Setup GPU execution grid
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + 15) / 16, (NY + 15) / 16);

    // Example: Run 1000 steps
    for(int t = 0; t < 1000; t++) {
        lbm_step<<<numBlocks, threadsPerBlock>>>(d_f, d_f_next, NX, NY, TAU);
        
        // Pointer Swap (Double Buffering)
        float* temp = d_f;
        d_f = d_f_next;
        d_f_next = temp;
    }

    // Copy back result to CPU if needed
    cudaMemcpy(h_f, d_f, total_size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(h_f);
    cudaFree(d_f);
    cudaFree(d_f_next);
    return 0;
}



void initialize_grid(float* f, int nx, int ny) {
    float rho0 = 1.0f;
    float ux0 = 0.0f;
    float uy0 = 0.0f;
    float u_sq = ux0 * ux0 + uy0 * uy0;

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int tid = y * nx + x;
            for (int i = 0; i < 9; i++) {
                float eu = h_cx[i] * ux0 + h_cy[i] * uy0;
                float feq = h_w[i] * rho0 * (1.0f + 3.0f*eu + 4.5f*eu*eu - 1.5f*u_sq);
                f[i * (nx * ny) + tid] = feq; 
            }
        }
    }
}


__global__ void lbm_step(float* d_fin, float* d_fout, int nx, int ny, float tau) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) return;

    int tid = y * nx + x;
    int grid_size = nx * ny;
    
    // 1. Calculate macroscopic Rho and Velocity
    float local_f[9];
    float rho = 0.0f;
    float ux_mom = 0.0f;
    float uy_mom = 0.0f;

    for(int i = 0; i < 9; i++) {
        local_f[i] = d_fin[i * grid_size + tid];
        rho += local_f[i];
        ux_mom += local_f[i] * d_cx[i];
        uy_mom += local_f[i] * d_cy[i];
    }

    float ux = ux_mom / rho;
    float uy = uy_mom / rho;
    float u_sq = ux * ux + uy * uy;

    // 2. Collision and Streaming
    for (int i = 0; i < 9; i++) {
        // Equilibrium
        float eu = d_cx[i] * ux + d_cy[i] * uy;
        float feq = d_w[i] * rho * (1.0f + 3.0f*eu + 4.5f*eu*eu - 1.5f*u_sq);

        // Relax (Collision)
        float f_new = local_f[i] - (1.0f / tau) * (local_f[i] - feq);

        // Find neighbor (Periodic Stream)
        int tx = (x + d_cx[i] + nx) % nx;
        int ty = (y + d_cy[i] + ny) % ny;
        int target_tid = ty * nx + tx;

        // Write to output stream
        d_fout[i * grid_size + target_tid] = f_new;
    }
}








