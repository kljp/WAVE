#include "comm.cuh"
#include "sab.cuh"

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void init_fqg(

        vertex_t src,
        depth_t *sa_d,
        const index_t *adj_deg_d,
        vertex_t *fq_td_d,
        vertex_t *fq_td_curr_sz
){

    depth_t depth = (depth_t) 0x00000000;
    index_t deg = adj_deg_d[src];
    depth_t fcls = sab<depth_t>::clfy_fcls(deg);

    sa_d[src] = sab<depth_t>::get_sab(fcls, depth);
    fq_td_d[*fq_td_curr_sz] = src;
    *fq_td_curr_sz = 1;
}

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void fqg_td_th(

        depth_t *sa_d,
        const vertex_t *adj_list_d,
        const index_t *offset_d,
        const index_t *adj_deg_d,
        const depth_t level,
        vertex_t *fq_td_d,
        vertex_t *fq_td_curr_sz,
        vertex_t *fq_td_th_d,
        vertex_t *fq_td_th_curr_sz
){

    index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const index_t grnt = blockDim.x * gridDim.x; // granularity
    const index_t fq_th_sz = (index_t) *fq_td_th_curr_sz;

    vertex_t vid;
    index_t deg_curr;
    index_t beg_pos;

    vertex_t nbid; // neighbor vertex id
    depth_t nb_sab_curr;
    index_t nb_deg_curr;

    while(tid < fq_th_sz){

        vid = fq_td_th_d[tid];
        deg_curr = adj_deg_d[vid];
        beg_pos = offset_d[vid];

        for(index_t i = 0; i < deg_curr; i++){

            nbid = adj_list_d[beg_pos + i];
////////////////////////////////// hub cache check (shared memory hub cahce scheme should be added)
            nb_sab_curr = sa_d[nbid];

            if(nb_sab_curr == SAB_INIT){

                nb_deg_curr = adj_deg_d[nbid];

                if(atomicCAS(&sa_d[nbid],
                             SAB_INIT,
                             sab<depth_t>::get_sab(sab<depth_t>::clfy_fcls(nb_deg_curr), (depth_t) (level + 1))
                             ) == SAB_INIT){

////////////////////////////////// hub cache replacement (shared memory hub cahce scheme should be added)
                    fq_td_d[atomicAdd(fq_td_curr_sz, 1)] = nbid;
                }
            }
        }

        tid += grnt;
    }
}

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void fqg_td_xw(

        depth_t *sa_d,
        const vertex_t *adj_list_d,
        const index_t *offset_d,
        const index_t *adj_deg_d,
        const depth_t level,
        vertex_t *fq_td_d,
        vertex_t *fq_td_curr_sz,
        vertex_t *fq_td_xw_d,
        vertex_t *fq_td_xw_curr_sz,
        const index_t th_x
){

    index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    index_t lid = tid % th_x; // laneID
    index_t xwid = tid / th_x; // warpID (uni-warp or mult-warp)
    const index_t grnt = blockDim.x * gridDim.x / th_x; // granularity
    const index_t fq_xw_sz = (index_t) *fq_td_xw_curr_sz;

    vertex_t vid;
    index_t deg_curr;
    index_t beg_pos;

    vertex_t nbid; // neighbor vertex id
    depth_t nb_sab_curr;
    index_t nb_deg_curr;

    while(xwid < fq_xw_sz){

        vid = fq_td_xw_d[xwid];
        deg_curr = adj_deg_d[vid];
        beg_pos = offset_d[vid];

        while(lid < deg_curr){

            nbid = adj_list_d[beg_pos + lid];
////////////////////////////////// hub cache check (shared memory hub cahce scheme should be added)
            nb_sab_curr = sa_d[nbid];

            if(nb_sab_curr == SAB_INIT){

                nb_deg_curr = adj_deg_d[nbid];

                if(atomicCAS(&sa_d[nbid],
                             SAB_INIT,
                             sab<depth_t>::get_sab(sab<depth_t>::clfy_fcls(nb_deg_curr), (depth_t) (level + 1))
                             ) == SAB_INIT){

////////////////////////////////// hub cache replacement (shared memory hub cahce scheme should be added)
                    fq_td_d[atomicAdd(fq_td_curr_sz, 1)] = nbid;
                }
            }

            lid += th_x;
        }

        xwid += grnt;
    }
}