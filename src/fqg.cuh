#include "comm.cuh"

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void init_fqg(

        vertex_t src,
        depth_t *sa_d,
        vertex_t *fq_td_d,
        vertex_t *fq_td_curr_sz
){

    sa_d[src] = (depth_t) 0x00000000;
    fq_td_d[*fq_td_curr_sz] = src;
    *fq_td_curr_sz = 1;
}

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void init_fqg_2(

        vertex_t *fq_td_d,
        vertex_t *fq_td_curr_sz
){

    fq_td_d[0] = -1;
    *fq_td_curr_sz = 0;
}

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void fqg_td_wccao( // warp-cooperative chained atomic operations

        depth_t *sa_d,
        const vertex_t *adj_list_d,
        const index_t *offset_d,
        const index_t *adj_deg_d,
        const depth_t level,
        vertex_t *fq_td_in_d,
        vertex_t *fq_td_in_curr_sz,
        vertex_t *fq_td_out_d,
        vertex_t *fq_td_out_curr_sz
){

    index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    index_t lid_st = tid % WSZ; // laneID
    index_t lid;
    index_t wid = tid / WSZ; // warpID
    const index_t grnt = blockDim.x * gridDim.x / WSZ; // granularity
    const index_t fq_sz = (index_t) *fq_td_in_curr_sz;

    vertex_t vid;
    index_t deg_curr;
    index_t beg_pos;

    vertex_t nbid; // neighbor vertex id
    depth_t nb_depth_curr;
    while(wid < fq_sz){

        vid = fq_td_in_d[wid];
        deg_curr = adj_deg_d[vid];
        beg_pos = offset_d[vid];
        lid = lid_st;

        while(lid < deg_curr){

            nbid = adj_list_d[beg_pos + lid];
            nb_depth_curr = sa_d[nbid];

            if(nb_depth_curr == INFTY){

                if(atomicCAS(&sa_d[nbid],
                             INFTY,
                             (depth_t) (level + 1)
                             ) == INFTY){

                    fq_td_out_d[atomicAdd(fq_td_out_curr_sz, 1)] = nbid;
                }
            }

            lid += WSZ;
        }

        wid += grnt;
    }
}

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void fqg_td_wcsac( // warp-cooperative status array check

        depth_t *sa_d,
        const vertex_t *adj_list_d,
        const index_t *offset_d,
        const index_t *adj_deg_d,
        const depth_t level,
        vertex_t *fq_td_in_d,
        vertex_t *fq_td_in_curr_sz
){

    index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    index_t lid_st = tid % WSZ; // laneID
    index_t lid;
    index_t wid = tid / WSZ; // warpID
    const index_t grnt = blockDim.x * gridDim.x / WSZ; // granularity
    const index_t fq_sz = (index_t) *fq_td_in_curr_sz;

    vertex_t vid;
    index_t deg_curr;
    index_t beg_pos;

    vertex_t nbid; // neighbor vertex id
    depth_t nb_depth_curr;
    while(wid < fq_sz){

        vid = fq_td_in_d[wid];
        deg_curr = adj_deg_d[vid];
        beg_pos = offset_d[vid];
        lid = lid_st;

        while(lid < deg_curr){

            nbid = adj_list_d[beg_pos + lid];
            nb_depth_curr = sa_d[nbid];

            if(nb_depth_curr == INFTY)
                sa_d[nbid] = (depth_t) (level + 1);

            lid += WSZ;
        }

        wid += grnt;
    }
}

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void fqg_td_tcfe( // thread-centric frontier enqueue

        depth_t *sa_d,
        const index_t vert_count,
        const depth_t level,
        vertex_t *fq_td_out_d,
        vertex_t *fq_td_out_curr_sz
){

    index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const index_t grnt = blockDim.x * gridDim.x; // granularity

    while(tid < vert_count){

        if(sa_d[tid] == level + 1)
            fq_td_out_d[atomicAdd(fq_td_out_curr_sz, 1)] = tid;

        tid += grnt;
    }
}

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void fqg_bu( // warp-cooperative neighbor check

        depth_t *sa_d,
        const vertex_t *adj_list_d,
        const index_t *offset_d,
        const index_t *adj_deg_d,
        const index_t vert_count,
        const depth_t level,
        vertex_t *success_bu_d,
        vertex_t *fq_bu_curr_sz
){

    index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    index_t lid_st = tid % WSZ; // laneID
    index_t lid;
    index_t wid = tid / WSZ; // warpID
    const index_t grnt = blockDim.x * gridDim.x / WSZ; // granularity

    index_t deg_curr;
    index_t beg_pos;

    vertex_t nbid; // neighbor vertex id
    index_t pred;

    if(lid_st == 0)
        success_bu_d[wid] = 0;

    while(wid < vert_count){

        if(sa_d[wid] == INFTY){

            deg_curr = adj_deg_d[wid];
            beg_pos = offset_d[wid];
            lid = lid_st;

            while(lid < deg_curr){

                pred = 0;
                nbid = adj_list_d[beg_pos + lid];

                if(sa_d[nbid] == level){

                    pred = 1;
                    sa_d[wid] = level + 1;
                }

                if(__ballot_sync(0xFFFFFFFF, pred) != 0){

                    if(lid_st == 0)
                        success_bu_d[wid]++;

                    break;
                }

                lid += WSZ;
            }
        }

        wid += grnt;
    }

    if(lid_st == 0)
        atomicAdd(fq_bu_curr_sz, success_bu_d[wid]);
}

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void fqg_rev_tcfe( // thread-centric frontier enqueue

        depth_t *sa_d,
        const index_t vert_count,
        const depth_t level,
        vertex_t *fq_td_in_d,
        vertex_t *fq_td_in_curr_sz
){

    index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const index_t grnt = blockDim.x * gridDim.x; // granularity

    while(tid < vert_count){

        if(sa_d[tid] == level)
            fq_td_in_d[atomicAdd(fq_td_in_curr_sz, 1)] = tid;

        tid += grnt;
    }
}

__global__ void asd(vertex_t *arr){
    *arr = 0;
}