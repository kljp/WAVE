#include "graph.h"
#include "alloc.cuh"
#include "wtime.h"
#include "comm.cuh"
#include "insp.cuh"
#include "fqg.cuh"

template<typename vertex_t, typename index_t, typename depth_t>
void bfs_td(

        depth_t *sa_d,
        const vertex_t *adj_list_d,
        const index_t *offset_d,
        const index_t *adj_deg_d,
        const index_t vert_count,
        cudaStream_t *stream,
        depth_t &level,
        vertex_t *fq_td_d,
        vertex_t *temp_fq_td_d,
        vertex_t *fq_td_curr_sz,
        vertex_t *temp_fq_td_curr_sz,
        vertex_t *fq_td_sz_h,
        vertex_t *fq_td_th_d,
        vertex_t *fq_td_th_curr_sz,
        vertex_t *fq_td_uw_d,
        vertex_t *fq_td_uw_curr_sz,
        vertex_t *fq_td_mw_d,
        vertex_t *fq_td_mw_curr_sz
){

    insp_clfy<vertex_t, index_t, depth_t>
    <<<BLKS_NUM, THDS_NUM>>>(

            sa_d,
            level,
            fq_td_d,
            fq_td_curr_sz,
            fq_td_th_d,
            fq_td_th_curr_sz,
            fq_td_uw_d,
            fq_td_uw_curr_sz,
            fq_td_mw_d,
            fq_td_mw_curr_sz
    );

    cudaDeviceSynchronize();

    // flush the frontier queue, if level == 0
    if(level != 0){

        H_ERR(cudaMemcpy(fq_td_d, temp_fq_td_d, sizeof(vertex_t) * vert_count, cudaMemcpyHostToDevice));
        H_ERR(cudaMemcpy(fq_td_curr_sz, temp_fq_td_curr_sz, sizeof(vertex_t), cudaMemcpyHostToDevice));
    }

    fqg_td_th<vertex_t, index_t, depth_t> // frontier queue generation for 'thread class'
    <<<BLKS_NUM, THDS_NUM, 0, stream[0]>>>(

            sa_d,
            adj_list_d,
            offset_d,
            adj_deg_d,
            level,
            fq_td_d,
            fq_td_curr_sz,
            fq_td_th_d,
            fq_td_th_curr_sz
    );

    fqg_td_xw<vertex_t, index_t, depth_t> // frontier queue generation for 'uni-warp class'
    <<<BLKS_NUM, THDS_NUM, 0, stream[1]>>>(

            sa_d,
            adj_list_d,
            offset_d,
            adj_deg_d,
            level,
            fq_td_d,
            fq_td_curr_sz,
            fq_td_uw_d,
            fq_td_uw_curr_sz,
            th_a
    );

    fqg_td_xw<vertex_t, index_t, depth_t> // frontier queue generation for 'mult-warp class'
    <<<BLKS_NUM, THDS_NUM, 0, stream[2]>>>(

            sa_d,
            adj_list_d,
            offset_d,
            adj_deg_d,
            level,
            fq_td_d,
            fq_td_curr_sz,
            fq_td_mw_d,
            fq_td_mw_curr_sz,
            th_b
    );

    for(index_t i = 0; i < Q_CARD; i++)
        cudaStreamSynchronize(stream[i]);

    H_ERR(cudaMemcpy(fq_td_sz_h, fq_td_curr_sz, sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

template<typename vertex_t, typename index_t, typename depth_t>
void bfs_tdbu(

        vertex_t src,
        depth_t *sa_d,
        const vertex_t *adj_list_d,
        const index_t *offset_d,
        const index_t *adj_deg_d,
        const index_t vert_count,
        cudaStream_t *stream,
        depth_t &level,
        vertex_t *fq_td_d,
        vertex_t *temp_fq_td_d,
        vertex_t *fq_td_curr_sz,
        vertex_t *temp_fq_td_curr_sz,
        vertex_t *fq_td_sz_h,
        vertex_t *fq_td_th_d,
        vertex_t *fq_td_th_curr_sz,
        vertex_t *fq_td_uw_d,
        vertex_t *fq_td_uw_curr_sz,
        vertex_t *fq_td_mw_d,
        vertex_t *fq_td_mw_curr_sz,
        vertex_t *fq_bu_d,
        vertex_t *temp_fq_bu_d,
        vertex_t *fq_bu_curr_sz,
        vertex_t *temp_fq_bu_curr_sz
){

    TD_BU = 0;

    for(level = 0; ; level++){

        for(index_t i = 0; i < Q_CARD; i++)
            cudaStreamSynchronize(stream[i]);

        std::cout << "level " << (int) level << std::endl;

        if(!TD_BU){

            H_ERR(cudaMemcpy(fq_td_th_d, temp_fq_td_d, sizeof(vertex_t) * vert_count, cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(fq_td_th_curr_sz, temp_fq_td_curr_sz, sizeof(vertex_t), cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(fq_td_uw_d, temp_fq_td_d, sizeof(vertex_t) * vert_count, cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(fq_td_uw_curr_sz, temp_fq_td_curr_sz, sizeof(vertex_t), cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(fq_td_mw_d, temp_fq_td_d, sizeof(vertex_t) * vert_count, cudaMemcpyHostToDevice));
            H_ERR(cudaMemcpy(fq_td_mw_curr_sz, temp_fq_td_curr_sz, sizeof(vertex_t), cudaMemcpyHostToDevice));

            if(level == 0){

                H_ERR(cudaMemcpy(fq_td_d, temp_fq_td_d, sizeof(vertex_t) * vert_count, cudaMemcpyHostToDevice));
                H_ERR(cudaMemcpy(fq_td_curr_sz, temp_fq_td_curr_sz, sizeof(vertex_t), cudaMemcpyHostToDevice));

                init_fqg<vertex_t, index_t, depth_t>
                <<<1, 1, 0, stream[0]>>>(

                        src,
                        sa_d,
                        adj_deg_d,
                        fq_td_d,
                        fq_td_curr_sz
                );

                for(index_t i = 0; i < Q_CARD; i++)
                    cudaStreamSynchronize(stream[i]);
            }

            std::cout << "Top-down phase" << std::endl;
            bfs_td<vertex_t, index_t, depth_t>(

                    sa_d,
                    adj_list_d,
                    offset_d,
                    adj_deg_d,
                    vert_count,
                    stream,
                    level,
                    fq_td_d,
                    temp_fq_td_d,
                    fq_td_curr_sz,
                    temp_fq_td_curr_sz,
                    fq_td_sz_h,
                    fq_td_th_d,
                    fq_td_th_curr_sz,
                    fq_td_uw_d,
                    fq_td_uw_curr_sz,
                    fq_td_mw_d,
                    fq_td_mw_curr_sz
            );
        }
        else{
            ;
        }

        if(!TD_BU){

            if(*fq_td_sz_h == 0)
                break;
        }
        else {
            ;
        }
    }
}

// Function called from CPU
template<typename vertex_t, typename index_t>
int bfs( // breadth-first search on GPU

        vertex_t src,
        index_t *beg_pos,
        vertex_t *csr,
        index_t vert_count,
        index_t edge_count,
        index_t gpu_id
){

    cudaSetDevice(gpu_id);

    depth_t *sa_d; // status array on GPU
    depth_t *sa_h; // status array on CPU
    depth_t *temp_sa; // initial state of status array (It will be used for iterative test in the future)
    index_t *adj_deg_d; // the number of neighbors for each vertex
    vertex_t *adj_list_d; // adjacent lists
    index_t *offset_d; // offset
    vertex_t *fq_td_d; // frontier queue for top-down traversal
    vertex_t *temp_fq_td_d; // used at the start of every level
    vertex_t *fq_td_curr_sz; // used for the top-down queue size
                            // synchronized index of frontier queue for top-down traversal, the size must be 1
    vertex_t *temp_fq_td_curr_sz;
    vertex_t *fq_td_sz_h;
    vertex_t *fq_td_th_d; // sub-queue for 'thread' frontier class
    vertex_t *fq_td_th_curr_sz;
    vertex_t *fq_td_uw_d; // sub-queue for 'uni-warp' frontier class
    vertex_t *fq_td_uw_curr_sz;
    vertex_t *fq_td_mw_d; // sub-queue for 'mult-warp' frontier class
    vertex_t *fq_td_mw_curr_sz;
    vertex_t *fq_bu_d; // frontier queue for bottom-up traversal
    vertex_t *temp_fq_bu_d; // used at the start of every iteration
    vertex_t *fq_bu_curr_sz; // used for the bottom-up queue size
                            // synchronized index of frontier queue for bottom-up traversal,
                            // the size must be 1, used for sync-ing atomic operations at the first level of bottom-up traversal
    vertex_t *temp_fq_bu_curr_sz;

    cudaStream_t *stream;

    alloc<vertex_t, index_t, depth_t>::
    alloc_mem(

            sa_d,
            sa_h,
            temp_sa,
            adj_list_d,
            adj_deg_d,
            offset_d,
            beg_pos,
            csr,
            vert_count,
            edge_count,
            stream,
            fq_td_d,
            temp_fq_td_d,
            fq_td_curr_sz,
            temp_fq_td_curr_sz,
            fq_td_sz_h,
            fq_td_th_d,
            fq_td_th_curr_sz,
            fq_td_uw_d,
            fq_td_uw_curr_sz,
            fq_td_mw_d,
            fq_td_mw_curr_sz,
            fq_bu_d,
            temp_fq_bu_d,
            fq_bu_curr_sz,
            temp_fq_bu_curr_sz
    );

    depth_t level;
    double t_st, t_end, t_elpd; // time_start, time_end, time_elapsed
//    double avg_teps = 0.0; // average_teps (traversed edges per second)
    double curr_teps = 0.0; // current_teps

    warm_up_gpu<<<BLKS_NUM, THDS_NUM>>>();
    cudaDeviceSynchronize();

    ///// iteration starts - currently only one iteration //////////////////////////////////////////////////////////////

    H_ERR(cudaMemcpy(sa_d, temp_sa, sizeof(depth_t) * vert_count, cudaMemcpyHostToDevice));
    level = 0;
    t_st = wtime();

    bfs_tdbu<vertex_t, index_t, depth_t>(

            src,
            sa_d,
            adj_list_d,
            offset_d,
            adj_deg_d,
            vert_count,
            stream,
            level,
            fq_td_d,
            temp_fq_td_d,
            fq_td_curr_sz,
            temp_fq_td_curr_sz,
            fq_td_sz_h,
            fq_td_th_d,
            fq_td_th_curr_sz,
            fq_td_uw_d,
            fq_td_uw_curr_sz,
            fq_td_mw_d,
            fq_td_mw_curr_sz,
            fq_bu_d,
            temp_fq_bu_d,
            fq_bu_curr_sz,
            temp_fq_bu_curr_sz
    );

    t_end = wtime();

    // for validation
    H_ERR(cudaMemcpy(sa_h, sa_d, sizeof(depth_t) * vert_count, cudaMemcpyDeviceToHost));
    index_t count = 0;
    for(index_t i = 0; i < vert_count; i++){
        if(sa_h[i] != SAB_INIT)
            count++;
    }
    std::cout << "The number of traversed vertices: " << count << std::endl;
    t_elpd = t_end - t_st;
    curr_teps = (double) edge_count / t_elpd;
    std::cout << "Consumed time (s): " << t_elpd << std::endl;
    std::cout << "Current TEPS: " << curr_teps << std::endl;

    ///// iteration ends ///////////////////////////////////////////////////////////////////////////////////////////////

    alloc<vertex_t, index_t, depth_t>::
    dealloc_mem(

            sa_d,
            sa_h,
            temp_sa,
            adj_list_d,
            adj_deg_d,
            offset_d,
            stream,
            fq_td_d,
            temp_fq_td_d,
            fq_td_curr_sz,
            temp_fq_td_curr_sz,
            fq_td_sz_h,
            fq_td_th_d,
            fq_td_th_curr_sz,
            fq_td_uw_d,
            fq_td_uw_curr_sz,
            fq_td_mw_d,
            fq_td_mw_curr_sz,
            fq_bu_d,
            temp_fq_bu_d,
            fq_bu_curr_sz,
            temp_fq_bu_curr_sz
    );

    std::cout << "GPU BFS finished" << std::endl;

    return 0;
}