#include "graph.h"
#include "alloc.cuh"
#include "wtime.h"
#include "comm.cuh"
#include "insp.cuh"
#include "fqg.cuh"
#include "mcpy.cuh"

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
        vertex_t *fq_td_mw_curr_sz,
        vertex_t *hub_hash
){

    insp_clfy<vertex_t, index_t, depth_t>
    <<<BLKS_NUM, THDS_NUM>>>(

            sa_d,
            fq_td_d,
            fq_td_curr_sz,
            fq_td_th_d,
            fq_td_th_curr_sz,
            fq_td_uw_d,
            fq_td_uw_curr_sz,
            fq_td_mw_d,
            fq_td_mw_curr_sz,
            hub_hash
    );

    cudaDeviceSynchronize();

    // flush the frontier queue
    if(level == 0){

        flush_fq<vertex_t, index_t, depth_t>
        <<<1,1>>>(

                fq_td_d,
                fq_td_curr_sz
        );
    }
    else{

        mcpy_init_fq_td<vertex_t, index_t, depth_t>
        <<<BLKS_NUM, THDS_NUM>>>(

                vert_count,
                temp_fq_td_d,
                temp_fq_td_curr_sz,
                fq_td_d,
                fq_td_curr_sz
        );
    }
    cudaDeviceSynchronize();

//    fqg_td_th<vertex_t, index_t, depth_t> // frontier queue generation for 'thread class'
//    <<<BLKS_NUM, THDS_NUM, 0, stream[0]>>>(
//
//            sa_d,
//            adj_list_d,
//            offset_d,
//            adj_deg_d,
//            level,
//            fq_td_d,
//            fq_td_curr_sz,
//            fq_td_th_d,
//            fq_td_th_curr_sz,
//            hub_hash
//    );
//    cudaDeviceSynchronize();

    fqg_td_xw<vertex_t, index_t, depth_t> // frontier queue generation for 'uni-warp class'
    <<<BLKS_NUM_UW, THDS_NUM_UW, 0, stream[1]>>>(

            sa_d,
            adj_list_d,
            offset_d,
            adj_deg_d,
            level,
            fq_td_d,
            fq_td_curr_sz,
            fq_td_uw_d,
            fq_td_uw_curr_sz,
            th_a,
            hub_hash
    );
    cudaDeviceSynchronize();

//    fqg_td_xw<vertex_t, index_t, depth_t> // frontier queue generation for 'mult-warp class'
//    <<<BLKS_NUM_MW, THDS_NUM_MW, 0, stream[2]>>>(
//
//            sa_d,
//            adj_list_d,
//            offset_d,
//            adj_deg_d,
//            level,
//            fq_td_d,
//            fq_td_curr_sz,
//            fq_td_mw_d,
//            fq_td_mw_curr_sz,
//            th_b,
//            hub_hash
//    );
//    cudaDeviceSynchronize();

//    for(index_t i = 0; i < Q_CARD; i++)
//        cudaStreamSynchronize(stream[i]);

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
        vertex_t *temp_fq_bu_curr_sz,
        vertex_t *hub_hash
){

    TD_BU = 0;

    for(level = 0; ; level++){

        for(index_t i = 0; i < Q_CARD; i++)
            cudaStreamSynchronize(stream[i]);

//        std::cout << "level " << (int) level << std::endl;

        if(!TD_BU){

            mcpy_init_fq_td<vertex_t, index_t, depth_t>
            <<<BLKS_NUM, THDS_NUM, 0, stream[0]>>>(

                    vert_count,
                    temp_fq_td_d,
                    temp_fq_td_curr_sz,
                    fq_td_th_d,
                    fq_td_th_curr_sz
            );

            mcpy_init_fq_td<vertex_t, index_t, depth_t>
            <<<BLKS_NUM, THDS_NUM, 0, stream[1]>>>(

                    vert_count,
                    temp_fq_td_d,
                    temp_fq_td_curr_sz,
                    fq_td_uw_d,
                    fq_td_uw_curr_sz
            );

            mcpy_init_fq_td<vertex_t, index_t, depth_t>
            <<<BLKS_NUM, THDS_NUM, 0, stream[2]>>>(

                    vert_count,
                    temp_fq_td_d,
                    temp_fq_td_curr_sz,
                    fq_td_mw_d,
                    fq_td_mw_curr_sz
            );

            for(index_t i = 0; i < Q_CARD; i++)
                cudaStreamSynchronize(stream[i]);

            if(level == 0){

                mcpy_init_fq_td<vertex_t, index_t, depth_t>
                <<<BLKS_NUM, THDS_NUM>>>(

                        vert_count,
                        temp_fq_td_d,
                        temp_fq_td_curr_sz,
                        fq_td_d,
                        fq_td_curr_sz
                );
                cudaDeviceSynchronize();

                init_fqg<vertex_t, index_t, depth_t>
                <<<1, 1>>>(

                        src,
                        sa_d,
                        adj_deg_d,
                        fq_td_d,
                        fq_td_curr_sz
                );
                cudaDeviceSynchronize();
            }

//            std::cout << "Top-down phase" << std::endl;
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
                    fq_td_mw_curr_sz,
                    hub_hash
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

        vertex_t *src_list,
        index_t *beg_pos,
        vertex_t *csr,
        index_t vert_count,
        index_t edge_count,
        index_t gpu_id
){

    cudaSetDevice(gpu_id);
//    copy_sm_sz<<<1, 1>>>(calc_sm_sz(gpu_id));
//    cudaDeviceSynchronize();

    depth_t *sa_d; // status array on GPU
    depth_t *sa_h; // status array on CPU
    depth_t *temp_sa; // initial state of status array (It will be used for iterative test in the future)
    index_t *adj_deg_d; // the number of neighbors for each vertex
    index_t *adj_deg_h;
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
    vertex_t *hub_hash;
    vertex_t *temp_hub_hash;

    cudaStream_t *stream;

    alloc<vertex_t, index_t, depth_t>::
    alloc_mem(

            sa_d,
            sa_h,
            temp_sa,
            adj_list_d,
            adj_deg_d,
            adj_deg_h,
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
            temp_fq_bu_curr_sz,
            hub_hash,
            temp_hub_hash
    );

    mcpy_init_temp<vertex_t, index_t, depth_t>
    <<<BLKS_NUM, THDS_NUM>>>(

            vert_count,
            temp_fq_td_d,
            temp_fq_td_curr_sz,
            temp_fq_bu_d,
            temp_fq_bu_curr_sz,
            temp_hub_hash
    );
    cudaDeviceSynchronize();

    depth_t level;
    double t_st, t_end, t_elpd; // time_start, time_end, time_elapsed
    double avg_teps = 0.0; // average_teps (traversed edges per second)
    double curr_teps; // current_teps

    warm_up_gpu<<<BLKS_NUM, THDS_NUM>>>();
    cudaDeviceSynchronize();

    ///// iteration starts - currently only one iteration //////////////////////////////////////////////////////////////

    for(index_t i = 0; i < NUM_ITER; i++){
        H_ERR(cudaMemcpy(sa_d, temp_sa, sizeof(depth_t) * vert_count, cudaMemcpyHostToDevice));
        H_ERR(cudaMemcpy(sa_h, temp_sa, sizeof(depth_t) * vert_count, cudaMemcpyHostToHost));
        H_ERR(cudaMemcpy(hub_hash, temp_hub_hash, sizeof(vertex_t) * HUB_SZ, cudaMemcpyDeviceToDevice));

        level = 0;
        std::cout << "<<Iteration " << i << ">>" << std::endl;
        std::cout << "Started from " << src_list[i] << std::endl;
        t_st = wtime();

        bfs_tdbu<vertex_t, index_t, depth_t>(

                src_list[i],
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
                temp_fq_bu_curr_sz,
                hub_hash
        );

        t_end = wtime();

        // for validation
        index_t tr_vert = 0;
        index_t tr_edge = 0;

        H_ERR(cudaMemcpy(sa_h, sa_d, sizeof(depth_t) * vert_count, cudaMemcpyDeviceToHost));

        for(index_t j = 0; j < vert_count; j++){
            if(sa_h[j] != SAB_INIT){

                tr_vert++;
                tr_edge += adj_deg_h[j];
            }
        }
        std::cout << "The number of traversed vertices: " << tr_vert << std::endl;
        std::cout << "The number of traversed edges: " << tr_edge << std::endl;
        t_elpd = t_end - t_st;
        curr_teps = (double) tr_edge / t_elpd;
        avg_teps += curr_teps;
        std::cout << "Consumed time (s): " << t_elpd << std::endl;
        std::cout << "Current TEPS (biliion): " << curr_teps / 1000000000 << std::endl;
    }

    avg_teps /= NUM_ITER;
    std::cout << "Average TEPS (biliion): " << avg_teps / 1000000000 << std::endl;

    ///// iteration ends ///////////////////////////////////////////////////////////////////////////////////////////////

    alloc<vertex_t, index_t, depth_t>::
    dealloc_mem(

            sa_d,
            sa_h,
            temp_sa,
            adj_list_d,
            adj_deg_d,
            adj_deg_h,
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
            temp_fq_bu_curr_sz,
            hub_hash,
            temp_hub_hash
    );

    std::cout << "GPU BFS finished" << std::endl;

    return 0;
}