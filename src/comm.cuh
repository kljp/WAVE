#ifndef __H_COMM__
#define __H_COMM__

#include <stdio.h>

typedef int vertex_t;
typedef int index_t;
typedef unsigned int depth_t;

index_t TD_BU; // 0: top-down, 1: bottom-up

const index_t BLKS_NUM_INIT = 4096;
const index_t THDS_NUM_INIT =  256;
const index_t BLKS_NUM_INIT_RT = 4096;
const index_t THDS_NUM_INIT_RT =  256;
const index_t BLKS_NUM_TD_WCCAO = 128;
const index_t THDS_NUM_TD_WCCAO =  256;
const index_t BLKS_NUM_TD_WCSAC = 32768;
const index_t THDS_NUM_TD_WCSAC =  256;
const index_t BLKS_NUM_TD_TCFE = 16384;
const index_t THDS_NUM_TD_TCFE =  256;
const index_t BLKS_NUM_BU_WCSA = 32768;
const index_t THDS_NUM_BU_WCSA =  256;
const index_t BLKS_NUM_REV_TCFE = 16384;
const index_t THDS_NUM_REV_TCFE =  256;

const float par_alpha = 0.00023;
const float par_beta = 0.137;

#define WSZ 32 // warp size
#define MAX_THDS_PER_BLKS 1024
#define MAX_THDS_RD (1024 * 1024)
__device__ const unsigned int WARPS_NUM_BU = THDS_NUM_BU_WCSA * BLKS_NUM_BU_WCSA / WSZ;

#define NUM_ITER 64
#define INFTY (unsigned int) (0xFFFFFFFF)

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

#endif