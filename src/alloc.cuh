#include "comm.cuh"

template<typename vertex_t, typename index_t, typename depth_t>
struct alloc{

    inline static __host__ void alloc_mem(

        depth_t* &sa_d, // status array
        depth_t* &sa_h,
        depth_t* &temp_sa,
        vertex_t* &adj_list_d, // adjacent lists
        index_t* &adj_deg_d, // the number of neigbors for each vertex
        index_t* &adj_deg_h,
        index_t* &offset_d, // offset
        index_t* beg_pos, // csr - offset
        vertex_t* csr, // csr - edges
        index_t vert_count, // the number of vertices
        index_t edge_count, // the number of edges
        cudaStream_t* &stream,
        vertex_t* &fq_td_d,
        vertex_t* &temp_fq_td_d,
        vertex_t* &fq_td_curr_sz,
        vertex_t* &temp_fq_td_curr_sz,
        vertex_t* &fq_td_sz_h,
        vertex_t* &fq_td_th_d,
        vertex_t* &fq_td_th_curr_sz,
        vertex_t* &fq_td_uw_d,
        vertex_t* &fq_td_uw_curr_sz,
        vertex_t* &fq_td_mw_d,
        vertex_t* &fq_td_mw_curr_sz,
        vertex_t* &fq_bu_d,
        vertex_t* &temp_fq_bu_d,
        vertex_t* &fq_bu_curr_sz,
        vertex_t* &temp_fq_bu_curr_sz,
        vertex_t* &hub_hash_vid,
        vertex_t* &temp_hub_hash_vid
    ){

        long cpu_bytes = 0;
        long gpu_bytes = 0;

        H_ERR(cudaMalloc((void **) &sa_d, sizeof(depth_t) * vert_count));
        gpu_bytes += sizeof(depth_t) * vert_count;

        index_t *temp_os = new index_t[vert_count];
        
        for(index_t i = 0; i < vert_count; i++)
            temp_os[i] = beg_pos[i + 1] - beg_pos[i];

        H_ERR(cudaMalloc((void **) &adj_deg_d, sizeof(index_t) * vert_count));
        H_ERR(cudaMemcpy(adj_deg_d, temp_os, sizeof(index_t) * vert_count, cudaMemcpyHostToDevice));
        gpu_bytes += sizeof(index_t) * vert_count;
        H_ERR(cudaMallocHost((void **) &adj_deg_h, sizeof(index_t) * vert_count));
        H_ERR(cudaMemcpy(adj_deg_h, temp_os, sizeof(index_t) * vert_count, cudaMemcpyHostToHost));
        cpu_bytes += sizeof(index_t) * vert_count;
        delete[] temp_os;

        H_ERR(cudaMalloc((void **) &offset_d, sizeof(index_t) * vert_count));
        H_ERR(cudaMemcpy(offset_d, beg_pos, sizeof(index_t) * vert_count, cudaMemcpyHostToDevice));
        gpu_bytes += sizeof(index_t) * vert_count;

        H_ERR(cudaMalloc((void **) &adj_list_d, sizeof(vertex_t) * edge_count));
        H_ERR(cudaMemcpy(adj_list_d, csr, sizeof(vertex_t) * edge_count, cudaMemcpyHostToDevice));
        gpu_bytes += sizeof(vertex_t) * edge_count;

        H_ERR(cudaMallocHost((void **) &temp_sa, sizeof(depth_t) * vert_count));
        for (index_t i = 0; i < vert_count; i++)
            temp_sa[i] = SAB_INIT;
        H_ERR(cudaMallocHost((void **) &sa_h, sizeof(depth_t) * vert_count));
        cpu_bytes += sizeof(depth_t) * vert_count * 2;

        H_ERR(cudaMalloc((void **) &fq_td_d, sizeof(vertex_t) * vert_count));
        gpu_bytes += sizeof(vertex_t) * vert_count;
        H_ERR(cudaMallocHost((void **) &temp_fq_td_d, sizeof(vertex_t) * vert_count));
        for (vertex_t i = 0; i < vert_count; i++)
            temp_fq_td_d[i] = -1;
        cpu_bytes += sizeof(vertex_t) * vert_count;
        H_ERR(cudaMalloc((void **) &fq_td_curr_sz, sizeof(vertex_t)));
        gpu_bytes += sizeof(vertex_t);
        H_ERR(cudaMallocHost((void **) &temp_fq_td_curr_sz, sizeof(vertex_t)));
        *temp_fq_td_curr_sz = 0;
        cpu_bytes += sizeof(vertex_t);
        H_ERR(cudaMallocHost((void **) &fq_td_sz_h, sizeof(vertex_t)));
        cpu_bytes += sizeof(vertex_t);
        H_ERR(cudaMalloc((void **) &fq_td_th_d, sizeof(vertex_t) * vert_count));
        gpu_bytes += sizeof(vertex_t) * vert_count;
        H_ERR(cudaMalloc((void **) &fq_td_th_curr_sz, sizeof(vertex_t)));
        gpu_bytes += sizeof(vertex_t);
        H_ERR(cudaMalloc((void **) &fq_td_uw_d, sizeof(vertex_t) * vert_count));
        gpu_bytes += sizeof(vertex_t) * vert_count;
        H_ERR(cudaMalloc((void **) &fq_td_uw_curr_sz, sizeof(vertex_t)));
        gpu_bytes += sizeof(vertex_t);
        H_ERR(cudaMalloc((void **) &fq_td_mw_d, sizeof(vertex_t) * vert_count));
        gpu_bytes += sizeof(vertex_t) * vert_count;
        H_ERR(cudaMalloc((void **) &fq_td_mw_curr_sz, sizeof(vertex_t)));
        gpu_bytes += sizeof(vertex_t);
        H_ERR(cudaMalloc((void **) &fq_bu_d, sizeof(vertex_t) * vert_count));
        gpu_bytes += sizeof(vertex_t) * vert_count;
        H_ERR(cudaMallocHost((void **) &temp_fq_bu_d, sizeof(vertex_t) * vert_count));
        for (vertex_t i = 0; i < vert_count; i++)
            temp_fq_bu_d[i] = -1;
        H_ERR(cudaMalloc((void **) &fq_bu_curr_sz, sizeof(vertex_t)));
        gpu_bytes += sizeof(vertex_t);
        H_ERR(cudaMallocHost((void **) &temp_fq_bu_curr_sz, sizeof(vertex_t)));
        *temp_fq_bu_curr_sz = 0;
        cpu_bytes += sizeof(vertex_t);
        H_ERR(cudaMalloc((void **) &hub_hash_vid, sizeof(vertex_t) * HUB_SZ));
        gpu_bytes += sizeof(vertex_t) * HUB_SZ;
        H_ERR(cudaMallocHost((void **) &temp_hub_hash_vid, sizeof(vertex_t) * HUB_SZ));
        for (vertex_t i = 0; i < HUB_SZ; i++)
            temp_hub_hash_vid[i] = -1;
        cpu_bytes += sizeof(vertex_t) * HUB_SZ;

        stream = (cudaStream_t *) malloc(sizeof(cudaStream_t) * Q_CARD);
        for(index_t i = 0; i < Q_CARD; i++)
            cudaStreamCreate(&(stream[i]));

        std::cout << "CPU alloc space: " << cpu_bytes << " bytes" << std::endl;
        std::cout << "GPU alloc space: " << gpu_bytes << " bytes" << std::endl;
    }

    inline static __host__ void dealloc_mem(
        
        depth_t* &sa_d,
        depth_t* &sa_h,
        depth_t* &temp_sa,
        vertex_t* &adj_list_d,
        index_t* &adj_deg_d,
        index_t* &adj_deg_h,
        index_t* &offset_d,
        cudaStream_t* &stream,
        vertex_t* &fq_td_d,
        vertex_t* &temp_fq_td_d,
        vertex_t* &fq_td_curr_sz,
        vertex_t* &temp_fq_td_curr_sz,
        vertex_t* &fq_td_sz_h,
        vertex_t* &fq_td_th_d,
        vertex_t* &fq_td_th_curr_sz,
        vertex_t* &fq_td_uw_d,
        vertex_t* &fq_td_uw_curr_sz,
        vertex_t* &fq_td_mw_d,
        vertex_t* &fq_td_mw_curr_sz,
        vertex_t* &fq_bu_d,
        vertex_t* &temp_fq_bu_d,
        vertex_t* &fq_bu_curr_sz,
        vertex_t* &temp_fq_bu_curr_sz,
        vertex_t* &hub_hash_vid,
        vertex_t* &temp_hub_hash_vid
    ){

        cudaFree(sa_d);
        cudaFree(sa_h);
        cudaFree(temp_sa);
        cudaFree(adj_list_d);
        cudaFree(adj_deg_d);
        cudaFree(adj_deg_h);
        cudaFree(offset_d);
        cudaFree(fq_td_d);
        cudaFree(temp_fq_td_d);
        cudaFree(fq_td_curr_sz);
        cudaFree(temp_fq_td_curr_sz);
        cudaFree(fq_td_sz_h);
        cudaFree(fq_td_th_d);
        cudaFree(fq_td_th_curr_sz);
        cudaFree(fq_td_uw_d);
        cudaFree(fq_td_uw_curr_sz);
        cudaFree(fq_td_mw_d);
        cudaFree(fq_td_mw_curr_sz);
        cudaFree(fq_bu_d);
        cudaFree(temp_fq_bu_d);
        cudaFree(fq_bu_curr_sz);
        cudaFree(temp_fq_bu_curr_sz);
        cudaFree(hub_hash_vid);
        cudaFree(temp_hub_hash_vid);

        free(stream);
    }
};