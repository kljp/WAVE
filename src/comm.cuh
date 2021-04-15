#ifndef __H_COMM__
#define __H_COMM__

#include <stdio.h>

typedef int vertex_t;
typedef int index_t;
typedef unsigned int depth_t;

index_t TD_BU; // 0: top-down, 1: bottom-up

const index_t th_a = 32; // threshold alpha
const index_t th_b = 1024; // threshold beta, 32 * 32

const index_t THDS_NUM =  256; // block dimension
const index_t  BLKS_NUM = 256; // grid dimension

__device__ vertex_t hub_sz_d;
vertex_t hub_sz;

#define Q_CARD 3

#define FCLS_TH (unsigned int) (0x00000003)
#define FCLS_UW (unsigned int) (0x00000002)
#define FCLS_MW (unsigned int) (0x00000001)

#define SAB_INIT (unsigned int) (0x3FFFFFFF)

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", \
        cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define H_ERR( err ) \
  (HandleError( err, __FILE__, __LINE__ ))

__global__ void warm_up_gpu(){

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float va, vb, vc;
    va = 0.1f;
    vb = 0.2f;

    for(int i = 0; i < 10; i++)
        vc += ((float) tid + va * vb);
}

vertex_t calc_sm_sz(index_t gpu_id){

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpu_id);
    unsigned int sm_sz = (unsigned int) deviceProp.sharedMemPerMultiprocessor;
    unsigned int wp_smx = (unsigned int) deviceProp.maxThreadsPerMultiProcessor / th_a;
    unsigned int wp_blk = THDS_NUM / th_a;
    unsigned int blk_smx = wp_smx / wp_blk;
    unsigned int sm_sz_blk = sm_sz / blk_smx;
    hub_sz = ((vertex_t) sm_sz_blk) / ((vertex_t) sizeof(vertex_t));
    hub_sz /= 2; // because we use two arrays for hub cache (vertex id array and status bit array)

    return hub_sz;
}

__global__ void copy_sm_sz(vertex_t hub_sz){

    hub_sz_d = hub_sz;
}

__global__ void flush_fq(vertex_t *fq_td_d, vertex_t *fq_td_curr_sz){

    fq_td_d[0] = -1;
    *fq_td_curr_sz = 0;
}

#endif