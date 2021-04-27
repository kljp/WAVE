#ifndef __H_COMM__
#define __H_COMM__

#include <stdio.h>

typedef int vertex_t;
typedef int index_t;
typedef unsigned int depth_t;

index_t TD_BU; // 0: top-down, 1: bottom-up

const index_t THDS_NUM =  96; // block dimension
const index_t  BLKS_NUM = 64; // grid dimension
const index_t THDS_NUM_FQG =  1024; // block dimension
const index_t  BLKS_NUM_FQG = 4096; // grid dimension

#define WSZ 32; // warp size

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