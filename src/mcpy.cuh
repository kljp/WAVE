#ifndef __H_MCPY__
#define __H_MCPY__

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void mcpy_init_temp(

        index_t vert_count,
        vertex_t *temp_fq_td_d,
        vertex_t *temp_fq_curr_sz
){

    index_t tid_st = threadIdx.x + blockDim.x * blockIdx.x;
    index_t tid;
    const index_t grnt = gridDim.x * blockDim.x;

    tid = tid_st;
    while(tid < vert_count){

        temp_fq_td_d[tid] = -1;
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
        vertex_t *fq_td_curr_sz
){

    index_t tid_st = threadIdx.x + blockDim.x * blockIdx.x;
    index_t tid;
    const index_t grnt = gridDim.x * blockDim.x;

    tid = tid_st;
    while(tid < vert_count){

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

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void calc_rmnd( // calculate remaining vertices

        vertex_t *fq_bu_curr_sz,
        const vertex_t * __restrict__ proc_bu
){
    printf("%d ", *fq_bu_curr_sz);
    printf("%d ", *proc_bu);
    printf("%d\n", *fq_bu_curr_sz = - *proc_bu);
    *fq_bu_curr_sz = *fq_bu_curr_sz - *proc_bu;
}

#endif // __H_MCPY__
