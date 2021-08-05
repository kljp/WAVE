#ifndef __H_COMM__
#define __H_COMM__

#include <stdio.h>

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

#define WSZ 32 // warp size

#define NUM_ITER 1024
#define UNVISITED (unsigned int) (0xFFFFFFFF)

double avg_deg;
double prob_high;
double par_alpha;
double par_beta;

bool verbose;

double t_fqg_td_wccao = 0.0;
double t_fqg_td_wcsac = 0.0;
double t_fqg_td_tcfe = 0.0;
double t_fqg_bu_wcsac = 0.0;
double t_fqg_rev_tcfe = 0.0;

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

template<typename vertex_t, typename index_t>
void calc_par_opt(

        const vertex_t * __restrict__  adj_deg_h,
        const index_t vert_count,
        const index_t edge_count
){

    avg_deg = (double) edge_count / vert_count;
    vertex_t cnt_high = 0;
    vertex_t sample_sz = vert_count / (1 + vert_count * 0.05 * 0.05);
    for(vertex_t i = 0; i < sample_sz; i++){
        if(adj_deg_h[rand() % vert_count] > avg_deg)
            cnt_high ++;
    }

    prob_high = (double) cnt_high / sample_sz;
    double base_alpha = avg_deg * prob_high;
    if(base_alpha > 1.0){
        double num_alpha = (double) vert_count * prob_high;
        double w_alpha_0 = log(num_alpha) / log(base_alpha);
        double w_alpha_1 = 32.0;
        par_alpha = w_alpha_0 / w_alpha_1;
    }
    else
        par_alpha = 1.0;

    double prob_low = 1.0 - prob_high;
    double base_beta = avg_deg * prob_low;
    double num_beta = (double) vert_count * prob_low;
    double w_beta_0 = log(num_beta) / log(base_beta);
    double w_beta_1 = 32.0;
    par_beta = abs(w_beta_0) / (w_beta_1 * avg_deg * avg_deg);
}

#endif