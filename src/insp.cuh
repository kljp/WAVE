#include "comm.cuh"
#include "sab.cuh"

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void insp_clfy(

        const depth_t *sa_d,
        vertex_t *fq_td_d,
        vertex_t *fq_td_curr_sz,
        vertex_t *fq_td_th_d,
        vertex_t *fq_td_th_curr_sz,
        vertex_t *fq_td_uw_d,
        vertex_t *fq_td_uw_curr_sz,
        vertex_t *fq_td_mw_d,
        vertex_t *fq_td_mw_curr_sz,
        vertex_t *hub_hash
){

    index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const index_t grnt = blockDim.x * gridDim.x; // granularity
    const index_t fq_sz = (index_t) *fq_td_curr_sz;

    vertex_t vid;
    depth_t sab_curr;
    depth_t fcls_curr;

    while(tid < fq_sz){

        vid = fq_td_d[tid];
        sab_curr = sa_d[vid];
        fcls_curr = sab<depth_t>::get_fcls(sab_curr);

        fq_td_uw_d[atomicAdd(fq_td_uw_curr_sz, 1)] = vid;
//        if(fcls_curr == FCLS_TH)
//            fq_td_th_d[atomicAdd(fq_td_th_curr_sz, 1)] = vid;
//
//        else if(fcls_curr == FCLS_MW){
//
//            fq_td_mw_d[atomicAdd(fq_td_mw_curr_sz, 1)] = vid;
////            hub_hash[vid % HUB_SZ] = vid;
//        }
//
//        else{
//
//            fq_td_uw_d[atomicAdd(fq_td_uw_curr_sz, 1)] = vid;
////            hub_hash[vid % HUB_SZ] = vid;
//        }

        tid += grnt;
    }
}