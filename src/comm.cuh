#ifndef __H_COMM__
#define __H_COMM__

#include <stdio.h>

typedef int vertex_t;
typedef int index_t;
typedef unsigned int depth_t;

index_t TD_BU; // 0: top-down, 1: bottom-up

#define BLKS_NUM_INIT 4096
#define THDS_NUM_INIT 256
#define BLKS_NUM_INIT_RT 4096
#define THDS_NUM_INIT_RT 256
#define BLKS_NUM_TD_WCCAO 128
#define THDS_NUM_TD_WCCAO 256
#define BLKS_NUM_TD_WCSAC 32768
#define THDS_NUM_TD_WCSAC 256
#define BLKS_NUM_TD_TCFE 16384
#define THDS_NUM_TD_TCFE 256
#define BLKS_NUM_BU_WCSA 40960
#define THDS_NUM_BU_WCSA 256
#define BLKS_NUM_REV_TCFE 16384
#define THDS_NUM_REV_TCFE 256

#define PAR_ALPHA 0.00023
#define PAR_BETA 0.13

#define WSZ 32 // warp size

#define NUM_ITER 1024
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