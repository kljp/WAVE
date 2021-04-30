#ifndef __H_RDCE__
#define __H_RDCE__


__global__ void rdce_il (int *g_idata, int *g_odata, unsigned int n){ // reduce_interleaved

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx >= n) return;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){

        if (tid < stride)
            idata[tid] += idata[tid + stride];

        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void rdce_ur_2 (int *g_idata, int *g_odata, unsigned int n){ // reduce_unroll_2

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){

        if (tid < stride)
            idata[tid] += idata[tid + stride];

        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void rdce_ur_4 (int *g_idata, int *g_odata, unsigned int n){ // reduce_unroll_4

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    if (idx + 3 * blockDim.x < n){

        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){

        if (tid < stride)
            idata[tid] += idata[tid + stride];

        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void rdce_ur_wp_8 (int *g_idata, int *g_odata, unsigned int n){ // reduce_unroll_warp_8

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n){

        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1){

        if (tid < stride)
            idata[tid] += idata[tid + stride];

        __syncthreads();
    }

    if (tid < 32){

        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

#endif // __H_RDCE__