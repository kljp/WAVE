#ifndef __H_MCPY__
#define __H_MCPY__

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void mcpy_init_temp(

        index_t vert_count,
        vertex_t *temp_fq_td_d,
        vertex_t *temp_fq_curr_sz,
        vertex_t INFTY
){

    index_t tid_st = threadIdx.x + blockDim.x * blockIdx.x;
    index_t tid;
    const index_t grnt = gridDim.x * blockDim.x;

    tid = tid_st;
    while(tid < vert_count){

        temp_fq_td_d[tid] = INFTY;
        tid += grnt;
    }

    if(tid_st == 0)
        *temp_fq_curr_sz = 0;
}

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void mcpy_init_fq_td(

        index_t vert_count,
        vertex_t *temp_fq_td_d,
        vertex_t *temp_fq_curr_sz,
        vertex_t *fq_td_d,
        vertex_t *fq_td_curr_sz,
        vertex_t INFTY
){

    index_t tid_st = threadIdx.x + blockDim.x * blockIdx.x;
    index_t tid;
    const index_t grnt = gridDim.x * blockDim.x;

    tid = tid_st;
    while(tid < vert_count){

        if(fq_td_d[tid] == INFTY)
            break;
        
        fq_td_d[tid] = temp_fq_td_d[tid];
        tid += grnt;
    }

    if(tid_st == 0)
        *fq_td_curr_sz = *temp_fq_curr_sz;
}

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void flush_fq(

        vertex_t *fq_curr_sz
){

    *fq_curr_sz = 0;
}

#endif // __H_MCPY__
