#include "graph.h"
#include "alloc.cuh"
#include "wtime.h"
#include "comm.cuh"
#include "fqg.cuh"
#include "mcpy.cuh"

template<typename vertex_t, typename index_t, typename depth_t>
void bfs_td(

        depth_t *sa_d,
        const vertex_t * __restrict__ adj_list_d,
        const index_t * __restrict__ offset_d,
        const index_t * __restrict__ adj_deg_d,
        const index_t vert_count,
        depth_t &level,
        vertex_t *fq_td_in_d,
        vertex_t *fq_td_in_curr_sz,
        vertex_t *fq_sz_h,
        vertex_t *fq_td_out_d,
        vertex_t *fq_td_out_curr_sz
){

    if(*fq_sz_h < (vertex_t) (par_alpha * vert_count)){

        fqg_td_wccao<vertex_t, index_t, depth_t> // warp-cooperative chained atomic operations
        <<<BLKS_NUM_TD_WCCAO, THDS_NUM_TD_WCCAO>>>(

                sa_d,
                adj_list_d,
                offset_d,
                adj_deg_d,
                level,
                fq_td_in_d,
                fq_td_in_curr_sz,
                fq_td_out_d,
                fq_td_out_curr_sz
        );
        cudaDeviceSynchronize();
    }

    else{

        fqg_td_wcsac<vertex_t, index_t, depth_t> // warp-cooperative status array check
        <<<BLKS_NUM_TD_WCSAC, THDS_NUM_TD_WCSAC>>>(

                sa_d,
                adj_list_d,
                offset_d,
                adj_deg_d,
                level,
                fq_td_in_d,
                fq_td_in_curr_sz
        );
        cudaDeviceSynchronize();

        fqg_td_tcfe<vertex_t, index_t, depth_t> // thread-centric frontier enqueue
        <<<BLKS_NUM_TD_TCFE, THDS_NUM_TD_TCFE>>>(

                sa_d,
                vert_count,
                level,
                fq_td_out_d,
                fq_td_out_curr_sz
        );
        cudaDeviceSynchronize();
    }

    H_ERR(cudaMemcpy(fq_sz_h, fq_td_out_curr_sz, sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

template<typename vertex_t, typename index_t, typename depth_t>
void bfs_bu(

        depth_t *sa_d,
        const vertex_t * __restrict__ adj_list_d,
        const index_t * __restrict__ offset_d,
        const index_t * __restrict__ adj_deg_d,
        const index_t vert_count,
        depth_t &level,
        vertex_t *fq_sz_h,
        vertex_t *fq_bu_curr_sz
){

    fqg_bu_wcsa<vertex_t, index_t, depth_t>
    <<<BLKS_NUM_BU_WCSA, THDS_NUM_BU_WCSA>>>(

            sa_d,
            adj_list_d,
            offset_d,
            adj_deg_d,
            vert_count,
            level,
            fq_bu_curr_sz
    );
    cudaDeviceSynchronize();

    H_ERR(cudaMemcpy(fq_sz_h, fq_bu_curr_sz, sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

template<typename vertex_t, typename index_t, typename depth_t>
void bfs_rev(

        depth_t *sa_d,
        const index_t vert_count,
        depth_t &level,
        vertex_t *fq_sz_h,
        vertex_t *fq_td_1_d,
        vertex_t *fq_td_1_curr_sz
){

    fqg_rev_tcfe<vertex_t, index_t, depth_t> // thread-centric frontier enqueue
    <<<BLKS_NUM_REV_TCFE, THDS_NUM_REV_TCFE>>>(

            sa_d,
            vert_count,
            level,
            fq_td_1_d,
            fq_td_1_curr_sz
    );
    cudaDeviceSynchronize();

    H_ERR(cudaMemcpy(fq_sz_h, fq_td_1_curr_sz, sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

template<typename vertex_t, typename index_t, typename depth_t>
void bfs_tdbu(

        depth_t *sa_d,
        const vertex_t * __restrict__ adj_list_d,
        const index_t * __restrict__ offset_d,
        const index_t * __restrict__ adj_deg_d,
        const index_t vert_count,
        depth_t &level,
        vertex_t *fq_td_1_d,
        vertex_t *temp_fq_td_d,
        vertex_t *fq_td_1_curr_sz,
        vertex_t *temp_fq_curr_sz,
        vertex_t *fq_sz_h,
        vertex_t *fq_td_2_d,
        vertex_t *fq_td_2_curr_sz,
        vertex_t *fq_bu_curr_sz
){

    index_t fq_swap = 1;
    index_t reversed = 0;
    TD_BU = 0;

    *fq_sz_h = 1;

    for(level = 0; ; level++){

        if(*fq_sz_h < (vertex_t) (par_beta * vert_count)){

            if(TD_BU)
                reversed = 1;

            TD_BU = 0;
        }
        else
            TD_BU = 1;

        if(!TD_BU){

            if(fq_swap == 0)
                fq_swap = 1;
            else
                fq_swap = 0;

            if(level != 0){

                if(reversed == 0){

                    if(fq_swap == 0){

                        mcpy_init_fq_td<vertex_t, index_t, depth_t>
                        <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                                vert_count,
                                temp_fq_td_d,
                                temp_fq_curr_sz,
                                fq_td_2_d,
                                fq_td_2_curr_sz
                        );
                    }

                    else{

                        if(level == 1){

                            init_fqg_2<vertex_t, index_t, depth_t>
                            <<<1, 1>>>(

                                    fq_td_1_d,
                                    fq_td_1_curr_sz
                            );
                        }

                        else{

                            mcpy_init_fq_td<vertex_t, index_t, depth_t>
                            <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                                    vert_count,
                                    temp_fq_td_d,
                                    temp_fq_curr_sz,
                                    fq_td_1_d,
                                    fq_td_1_curr_sz
                            );
                        }
                    }
                }

                else{

                    reversed = 0;
                    fq_swap = 0;

                    mcpy_init_fq_td<vertex_t, index_t, depth_t>
                    <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                            vert_count,
                            temp_fq_td_d,
                            temp_fq_curr_sz,
                            fq_td_2_d,
                            fq_td_2_curr_sz
                    );

                    mcpy_init_fq_td<vertex_t, index_t, depth_t>
                    <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                            vert_count,
                            temp_fq_td_d,
                            temp_fq_curr_sz,
                            fq_td_1_d,
                            fq_td_1_curr_sz
                    );
                    cudaDeviceSynchronize();

                    bfs_rev<vertex_t, index_t, depth_t>(

                            sa_d,
                            vert_count,
                            level,
                            fq_sz_h,
                            fq_td_1_d,
                            fq_td_1_curr_sz
                    );
                }
            }

            cudaDeviceSynchronize();

            if(fq_swap == 0){

                bfs_td<vertex_t, index_t, depth_t>(

                        sa_d,
                        adj_list_d,
                        offset_d,
                        adj_deg_d,
                        vert_count,
                        level,
                        fq_td_1_d,
                        fq_td_1_curr_sz,
                        fq_sz_h,
                        fq_td_2_d,
                        fq_td_2_curr_sz
                );
            }

            else{

                bfs_td<vertex_t, index_t, depth_t>(

                        sa_d,
                        adj_list_d,
                        offset_d,
                        adj_deg_d,
                        vert_count,
                        level,
                        fq_td_2_d,
                        fq_td_2_curr_sz,
                        fq_sz_h,
                        fq_td_1_d,
                        fq_td_1_curr_sz
                );
            }
        }
        else{

            flush_fq<vertex_t, index_t, depth_t>
            <<<1, 1>>>(

                    fq_bu_curr_sz
            );
            cudaDeviceSynchronize();

            bfs_bu<vertex_t, index_t, depth_t>(

                    sa_d,
                    adj_list_d,
                    offset_d,
                    adj_deg_d,
                    vert_count,
                    level,
                    fq_sz_h,
                    fq_bu_curr_sz
            );
        }
        cudaDeviceSynchronize();

        if(!TD_BU){

            if(*fq_sz_h == 0)
                break;
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
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    depth_t *sa_d; // status array on GPU
    depth_t *sa_h; // status array on CPU
    depth_t *temp_sa; // initial state of status array (It will be used for iterative test in the future)
    index_t *adj_deg_d; // the number of neighbors for each vertex
    index_t *adj_deg_h;
    vertex_t *adj_list_d; // adjacent lists
    index_t *offset_d; // offset
    vertex_t *fq_td_1_d; // frontier queue for top-down traversal
    vertex_t *fq_td_1_curr_sz; // used for the top-down queue size
                            // synchronized index of frontier queue for top-down traversal, the size must be 1
    vertex_t *fq_td_2_d;
    vertex_t *fq_td_2_curr_sz;
    vertex_t *temp_fq_td_d;
    vertex_t *temp_fq_curr_sz;
    vertex_t *fq_sz_h;
    vertex_t *fq_bu_curr_sz; // used for the number of vertices examined at each level, the size must be 1

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
            fq_td_1_d,
            temp_fq_td_d,
            fq_td_1_curr_sz,
            temp_fq_curr_sz,
            fq_sz_h,
            fq_td_2_d,
            fq_td_2_curr_sz,
            fq_bu_curr_sz
    );

    mcpy_init_temp<vertex_t, index_t, depth_t>
    <<<BLKS_NUM_INIT, THDS_NUM_INIT>>>(

            vert_count,
            temp_fq_td_d,
            temp_fq_curr_sz
    );
    cudaDeviceSynchronize();

    depth_t level;
    double t_st, t_end, t_elpd; // time_start, time_end, time_elapsed
    double avg_gteps = 0.0; // average_teps (traversed edges per second)
    double curr_gteps; // current_teps

    warm_up_gpu<<<BLKS_NUM_INIT, THDS_NUM_INIT>>>();
    cudaDeviceSynchronize();

    ///// iteration starts /////////////////////////////////////////////////////////////////////////////////////////////

    for(index_t i = 0; i < NUM_ITER; i++){
        H_ERR(cudaMemcpy(sa_d, temp_sa, sizeof(depth_t) * vert_count, cudaMemcpyHostToDevice));
        H_ERR(cudaMemcpy(sa_h, temp_sa, sizeof(depth_t) * vert_count, cudaMemcpyHostToHost));

        mcpy_init_fq_td<vertex_t, index_t, depth_t>
        <<<BLKS_NUM_INIT, THDS_NUM_INIT>>>(

                vert_count,
                temp_fq_td_d,
                temp_fq_curr_sz,
                fq_td_1_d,
                fq_td_1_curr_sz
        );
        cudaDeviceSynchronize();

        mcpy_init_fq_td<vertex_t, index_t, depth_t>
        <<<BLKS_NUM_INIT, THDS_NUM_INIT>>>(

                vert_count,
                temp_fq_td_d,
                temp_fq_curr_sz,
                fq_td_2_d,
                fq_td_2_curr_sz
        );
        cudaDeviceSynchronize();

        init_fqg<vertex_t, index_t, depth_t>
        <<<1, 1>>>(

                src_list[i],
                sa_d,
                fq_td_1_d,
                fq_td_1_curr_sz
        );
        cudaDeviceSynchronize();

        level = 0;
        std::cout << "===========================================================" << std::endl;
        std::cout << "<<Iteration " << i << ">>" << std::endl;
        std::cout << "Started from " << src_list[i] << std::endl;
        t_st = wtime();

        bfs_tdbu<vertex_t, index_t, depth_t>(

                sa_d,
                adj_list_d,
                offset_d,
                adj_deg_d,
                vert_count,
                level,
                fq_td_1_d,
                temp_fq_td_d,
                fq_td_1_curr_sz,
                temp_fq_curr_sz,
                fq_sz_h,
                fq_td_2_d,
                fq_td_2_curr_sz,
                fq_bu_curr_sz
        );

        t_end = wtime();

        // for validation
        index_t tr_vert = 0;
        index_t tr_edge = 0;

        H_ERR(cudaMemcpy(sa_h, sa_d, sizeof(depth_t) * vert_count, cudaMemcpyDeviceToHost));

        for(index_t j = 0; j < vert_count; j++){
            if(sa_h[j] != INFTY){

                tr_vert++;
                tr_edge += adj_deg_h[j];
            }
        }

        std::cout << "The number of traversed vertices: " << tr_vert << std::endl;
        std::cout << "The number of traversed edges: " << tr_edge << std::endl;
        t_elpd = t_end - t_st;
        curr_gteps = (double) (tr_edge / t_elpd) / 1000000000;
        avg_gteps += curr_gteps;
        std::cout << "Consumed time (s): " << t_elpd << std::endl;
        std::cout << "Current GTEPS: " << curr_gteps << std::endl;
    }

    avg_gteps /= NUM_ITER;
    std::cout << "===========================================================" << std::endl;

    std::cout << "Average GTEPS: " << avg_gteps << std::endl;
    std::cout << "===========================================================" << std::endl;

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
            fq_td_1_d,
            temp_fq_td_d,
            fq_td_1_curr_sz,
            temp_fq_curr_sz,
            fq_sz_h,
            fq_td_2_d,
            fq_td_2_curr_sz,
            fq_bu_curr_sz
    );

    std::cout << "GPU BFS finished" << std::endl;

    return 0;
}