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

    double t_st;

    if(*fq_sz_h < (vertex_t) (par_alpha * vert_count)){

        if(verbose)
            t_st = wtime();
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
        if(verbose)
            t_fqg_td_wccao += (wtime() - t_st);
    }

    else{

        if(verbose)
            t_st = wtime();
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
        if(verbose)
            t_fqg_td_wcsac += (wtime() - t_st);

        if(verbose)
            t_st = wtime();
        fqg_td_tcfe<vertex_t, index_t, depth_t> // thread-centric frontier enqueue
        <<<BLKS_NUM_TD_TCFE, THDS_NUM_TD_TCFE>>>(

                sa_d,
                vert_count,
                level,
                fq_td_out_d,
                fq_td_out_curr_sz
        );
        cudaDeviceSynchronize();
        if(verbose)
            t_fqg_td_tcfe += (wtime() - t_st);
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

    double t_st;

    if(verbose)
        t_st = wtime();
    fqg_bu_wcsac<vertex_t, index_t, depth_t>
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
    if(verbose)
        t_fqg_bu_wcsac += (wtime() - t_st);

    H_ERR(cudaMemcpy(fq_sz_h, fq_bu_curr_sz, sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

template<typename vertex_t, typename index_t, typename depth_t>
void bfs_rev(

        depth_t *sa_d,
        const index_t vert_count,
        depth_t &level,
        vertex_t *fq_sz_h,
        vertex_t *fq_td_in_d,
        vertex_t *fq_td_in_curr_sz
){

    double t_st;

    if(verbose)
        t_st = wtime();
    fqg_rev_tcfe<vertex_t, index_t, depth_t> // thread-centric frontier enqueue
    <<<BLKS_NUM_REV_TCFE, THDS_NUM_REV_TCFE>>>(

            sa_d,
            vert_count,
            level,
            fq_td_in_d,
            fq_td_in_curr_sz
    );
    cudaDeviceSynchronize();
    if(verbose)
        t_fqg_rev_tcfe += (wtime() - t_st);

    H_ERR(cudaMemcpy(fq_sz_h, fq_td_in_curr_sz, sizeof(vertex_t), cudaMemcpyDeviceToHost));
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
        vertex_t *fq_bu_curr_sz,
        vertex_t INFTY
){

    bool fq_swap = true;
    bool reversed = false;
    bool TD_BU = false; // true: top-down, false: bottom-up

    *fq_sz_h = 1;

    for(level = 0; ; level++){

        if(*fq_sz_h < (vertex_t) (par_beta * vert_count)){

            if(TD_BU)
                reversed = true;

            TD_BU = false;
        }

        else
            TD_BU = true;


        if(!TD_BU){

            if(!fq_swap)
                fq_swap = true;
            else
                fq_swap = false;

            if(level != 0){

                if(!reversed){

                    if(!fq_swap){

                        mcpy_init_fq_td<vertex_t, index_t, depth_t>
                        <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                                vert_count,
                                temp_fq_td_d,
                                temp_fq_curr_sz,
                                fq_td_2_d,
                                fq_td_2_curr_sz,
                                INFTY
                        );
                    }

                    else{

                        if(level == 1){

                            init_fqg_2<vertex_t, index_t, depth_t>
                            <<<1, 1>>>(

                                    fq_td_1_d,
                                    fq_td_1_curr_sz,
                                    INFTY
                            );
                        }

                        else{

                            mcpy_init_fq_td<vertex_t, index_t, depth_t>
                            <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                                    vert_count,
                                    temp_fq_td_d,
                                    temp_fq_curr_sz,
                                    fq_td_1_d,
                                    fq_td_1_curr_sz,
                                    INFTY
                            );
                        }
                    }
                }

                else{

                    reversed = false;
                    fq_swap = false;

                    mcpy_init_fq_td<vertex_t, index_t, depth_t>
                    <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                            vert_count,
                            temp_fq_td_d,
                            temp_fq_curr_sz,
                            fq_td_2_d,
                            fq_td_2_curr_sz,
                            INFTY
                    );

                    mcpy_init_fq_td<vertex_t, index_t, depth_t>
                    <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                            vert_count,
                            temp_fq_td_d,
                            temp_fq_curr_sz,
                            fq_td_1_d,
                            fq_td_1_curr_sz,
                            INFTY
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

            if(!fq_swap){

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

            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
        }

        if(!TD_BU){

            if(*fq_sz_h == 0)
                break;
        }
    }
}

// Function called from CPU
template<typename vertex_t, typename index_t, typename depth_t>
int bfs( // breadth-first search on GPU

        vertex_t *src_list,
        index_t *beg_pos,
        vertex_t *csr,
        index_t vert_count,
        index_t edge_count,
        index_t gpu_id,
        vertex_t INFTY
){

    srand((unsigned int) wtime());
    int retry = 0;

    cudaSetDevice(gpu_id);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    depth_t *sa_d; // status array on GPU
    depth_t *sa_h; // status array on CPU
    depth_t *temp_sa; // initial state of status array (used for iterative test)
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
            temp_fq_curr_sz,
            INFTY
    );
    cudaDeviceSynchronize();

    depth_t level;
    double avg_prob_high = 0.0;
    double avg_depth = 0.0;
    double t_st, t_end, t_elpd, avg_t; // time_start, time_end, time_elapsed
    double t_par_st, t_par_end, t_par_elpd, avg_t_par; // time_calc_par_opt_start, time_calc_par_opt_end, time_calc_par_opt_elapsed
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
                fq_td_1_curr_sz,
                INFTY
        );
        cudaDeviceSynchronize();

        mcpy_init_fq_td<vertex_t, index_t, depth_t>
        <<<BLKS_NUM_INIT, THDS_NUM_INIT>>>(

                vert_count,
                temp_fq_td_d,
                temp_fq_curr_sz,
                fq_td_2_d,
                fq_td_2_curr_sz,
                INFTY
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

        if(!retry){

            std::cout << "===========================================================" << std::endl;
            std::cout << "<<Iteration " << i << ">>" << std::endl;
//        std::cout << "Started from " << src_list[i] << std::endl;
        }

        t_st = wtime();

        t_par_st = wtime();
        calc_par_opt<vertex_t, index_t>(

                adj_deg_h,
                vert_count,
                edge_count
        );
        t_par_end = wtime();

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
                fq_bu_curr_sz,
                INFTY
        );

        t_end = wtime();

        // for validation
        index_t tr_vert = 0;
        index_t tr_edge = 0;

        H_ERR(cudaMemcpy(sa_h, sa_d, sizeof(depth_t) * vert_count, cudaMemcpyDeviceToHost));

        for(index_t j = 0; j < vert_count; j++){
            if(sa_h[j] != UNVISITED){

                tr_vert++;
                tr_edge += adj_deg_h[j];
            }
        }

        // Retry the traversal due to bad source (the input graph is disconnected)
        if(tr_vert < (double) vert_count * 0.5 || tr_edge < (double) edge_count * 0.7){

            src_list[i] = rand() % vert_count;
            i--;
            retry++;
            continue;
        }
        retry = 0;

        std::cout << "Started from " << src_list[i] << std::endl;
        std::cout << "The number of traversed vertices: " << tr_vert << std::endl;
        std::cout << "The number of traversed edges: " << tr_edge << std::endl;
        avg_prob_high += prob_high;
        avg_depth += level;
        t_elpd = t_end - t_st;
        avg_t += t_elpd;
        t_par_elpd = (t_par_end - t_par_st) * 1000000.0;
        avg_t_par += t_par_elpd;
        curr_gteps = (double) (tr_edge / t_elpd) / 1000000000;
        avg_gteps += curr_gteps;
        std::cout << "The probability of high-degree vertex: " << prob_high << std::endl;
        std::cout << "Depth: " << level << std::endl;
        std::cout << "Overhead of direction-optimizer: " << t_par_elpd << "us" << std::endl;
        std::cout << "Consumed time: " << t_elpd << "s" << std::endl;
        std::cout << "Current GTEPS: " << curr_gteps << std::endl;
    }

    avg_prob_high /= NUM_ITER;
    avg_depth /= NUM_ITER;
    avg_t_par /= NUM_ITER;
    avg_t /= NUM_ITER;
    avg_gteps /= NUM_ITER;
    std::cout << "===================================================================" << std::endl;
    std::cout << "Summary of BFS" << std::endl;
    std::cout << "===================================================================" << std::endl;
    std::cout << "Average degree: " << avg_deg << std::endl;
    std::cout << "Average probability of high-degree vertex: " << avg_prob_high << std::endl;
    std::cout << "Average depth: " << avg_depth << std::endl;
    std::cout << "Average overhead of direction-optimizer: " << avg_t_par << "us" << std::endl;
    std::cout << "Average consumed time: " << avg_t << "s" << std::endl;
    std::cout << "Average GTEPS: " << avg_gteps << std::endl;
    std::cout << "===================================================================" << std::endl;
    if(verbose){

        t_fqg_td_wccao = (t_fqg_td_wccao * 1000000.0) / NUM_ITER;
        t_fqg_td_wcsac = (t_fqg_td_wcsac * 1000000.0) / NUM_ITER;
        t_fqg_td_tcfe = (t_fqg_td_tcfe * 1000000.0) / NUM_ITER;
        t_fqg_bu_wcsac = (t_fqg_bu_wcsac * 1000000.0) / NUM_ITER;
        t_fqg_rev_tcfe = (t_fqg_rev_tcfe * 1000000.0) / NUM_ITER;

        double t_wcfp_small = 0.0;
        double t_wcfp_medium = 0.0;
        double t_wcfp_large = 0.0;
        double t_wcfp_total = 0.0;
        t_wcfp_small = t_fqg_td_wccao;
        t_wcfp_medium = t_fqg_td_wcsac + t_fqg_td_tcfe;
        t_wcfp_large = t_fqg_bu_wcsac + t_fqg_rev_tcfe;
        t_wcfp_total = t_wcfp_small + t_wcfp_medium + t_wcfp_large;

        std::cout << "Breakdown of kernel execution of frontier processing techniques" << std::endl;
        std::cout << "===================================================================" << std::endl;
        std::cout << "fqg_td_wccao: " << t_fqg_td_wccao << "us (" << t_fqg_td_wccao * 100 / t_wcfp_total << "%)" << std::endl;
        std::cout << "fqg_td_wcsac: " << t_fqg_td_wcsac << "us (" << t_fqg_td_wcsac * 100 / t_wcfp_total << "%)" << std::endl;
        std::cout << "fqg_td_tcfe: " << t_fqg_td_tcfe << "us (" << t_fqg_td_tcfe * 100 / t_wcfp_total << "%)" << std::endl;
        std::cout << "fqg_bu_wcsac " << t_fqg_bu_wcsac << "us (" << t_fqg_bu_wcsac * 100 / t_wcfp_total << "%)" << std::endl;
        std::cout << "fqg_rev_tcfe: " << t_fqg_rev_tcfe << "us (" << t_fqg_rev_tcfe * 100 / t_wcfp_total << "%)" << std::endl;
        std::cout << "-------------------------------------------------------------------" << std::endl;
        std::cout << "WCFP_SMALL: " << t_wcfp_small << "us (" << t_wcfp_small * 100 / t_wcfp_total << "%)" << std::endl;
        std::cout << "WCFP_MEDIUM: " << t_wcfp_medium << "us (" << t_wcfp_medium * 100 / t_wcfp_total << "%)" << std::endl;
        std::cout << "WCFP_LARGE: " << t_wcfp_large << "us (" << t_wcfp_large * 100 / t_wcfp_total << "%)" << std::endl;
        std::cout << "-------------------------------------------------------------------" << std::endl;
        std::cout << "Top-down: " << t_wcfp_small + t_wcfp_medium << "us (" << (t_wcfp_small + t_wcfp_medium) * 100 / t_wcfp_total << "%)" << std::endl;
        std::cout << "Bottom-up: " << t_wcfp_large << "us (" << t_wcfp_large * 100 / t_wcfp_total << "%)" << std::endl;
        std::cout << "===================================================================" << std::endl;
        std::cout << "Parameters" << std::endl;
        std::cout << "alpha: " << par_alpha << std::endl;
        std::cout << "beta: " << par_beta << std::endl;
        std::cout << "===================================================================" << std::endl;
    }

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