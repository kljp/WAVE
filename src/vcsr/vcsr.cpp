#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <cstdio>
#include <climits>
#include "wtime.h"

template <typename data_t>
void preprocess_graph(

        std::string filename,
        bool is_undirected,
        bool is_virtual,
        bool is_sorted,
        bool is_verylarge,
        double t_elpd_st,
        int max_degree,
        data_t INFTY
){

    std::ifstream file;
    double t_st;

    bool fin_beg_pos = false;
    bool fin_adj_list = false;
    bool fin_vid_pos = false;
    bool fin_vid_list = false;
    bool fin_virt_beg_pos = false;

    file.open(filename.c_str(), std::ios::in);
    if(!file.is_open()){
        std::cout << "[Invalid] Cannot open file. Maybe the file does not exist." << std::endl;
        std::cout << "\nPlease make sure that input file exists." << std::endl;
        exit(-1);
    }
    std::cout << "[Valid] Successfully opened the file: " << filename << std::endl;

    std::string line;
    while(true){

        std::getline(file, line);
        if(line[0] != '%')
            break;
    }

    data_t disk_beg_pos, disk_adj_list, disk_vid_pos, disk_vid_list, disk_virt_beg_pos;
    data_t disk_acc = 0;

    data_t nodes_x, nodes_y, edges;
    int item_scanned;
    if(is_verylarge)
        item_scanned = sscanf(line.c_str(), "%llu %llu %llu", &nodes_x, &nodes_y, &edges);
    else
        item_scanned = sscanf(line.c_str(), "%u %u %u", &nodes_x, &nodes_y, &edges);
    if(item_scanned < 3 || nodes_x <= 0 || nodes_y <= 0 || edges <= 0){
        std::cout << "[Invalid] Graph metadata is not valid." << std::endl;
        std::cout << "\nPlease make sure that the first line after all comments represents nodes_x, node_y, and edges." << std::endl;
        exit(-1);
    }

    data_t vert_count = nodes_x;
    data_t edge_count = edges;
    std::cout << "[Info] NUM_VERTICES = " << vert_count << std::endl;
    std::cout << "[Info] NUM_EDGES = " << edge_count;
    if(is_undirected){
        edge_count *= 2;
        std::cout << " -> " << edge_count << " (doubling edges)";
    }
    std::cout << std::endl;

    data_t cursor = file.tellg();

    data_t v_cur, v_max, v_src, v_dest;
    v_max = 0;

    data_t num_loop = edge_count;
    if(is_undirected)
        num_loop /= 2;

    t_st = wtime();
    std::cout << "[Progress] Calculating v_max . . . ";
    for(data_t i = 0; i < num_loop; i++){

        std::getline(file, line);
        if(is_verylarge)
            item_scanned = sscanf(line.c_str(), "%llu %llu", &v_src, &v_dest);
        else
            item_scanned = sscanf(line.c_str(), "%u %u", &v_src, &v_dest);

        if(item_scanned < 2){
            std::cout << "[Invalid] An invalid edge that has neither v_src or v_dest was found." << std::endl;
            exit(-1);
        }

        v_cur = (v_src > v_dest) ? v_src : v_dest;
        if(v_max < v_cur)
            v_max = v_cur;
    }
    std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;

    data_t *degree = (data_t *) malloc(sizeof(data_t) * vert_count);
    data_t *mapping = (data_t *) malloc(sizeof(data_t) * (v_max + 1));

    for(data_t i = 0; i < vert_count; i++)
        degree[i] = 0;
    for(data_t i = 0; i < v_max + 1; i++)
        mapping[i] = INFTY;

    file.seekg(cursor);
    data_t cnt_mapping = 0;

    t_st = wtime();
    std::cout << "[Progress] Calculating degrees . . . ";
    for(data_t i = 0; i < num_loop; i++){

        std::getline(file, line);
        if(is_verylarge)
            sscanf(line.c_str(), "%llu %llu", &v_src, &v_dest);
        else
            sscanf(line.c_str(), "%u %u", &v_src, &v_dest);

        if(mapping[v_src] == INFTY){

            mapping[v_src] = cnt_mapping;
            cnt_mapping++;
        }
        degree[mapping[v_src]]++;

        if(is_undirected){

            if(mapping[v_dest] == INFTY){

                mapping[v_dest] = cnt_mapping;
                cnt_mapping++;
            }
            degree[mapping[v_dest]]++;
        }
    }
    std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;

    data_t *beg_pos = (data_t *) malloc(sizeof(data_t) * (vert_count + 1));
    data_t *adj_list = (data_t *) malloc(sizeof(data_t) * edge_count);

    t_st = wtime();
    std::cout << "[Progress] Constructing beginning position array (Offset Array) . . . ";
    beg_pos[0] = 0;
    beg_pos[vert_count] = edge_count;
    for(data_t i = 1; i < vert_count; i++){
        beg_pos[i] = beg_pos[i - 1] + degree[i - 1];
        degree[i - 1] = 0;
    }
    degree[vert_count - 1] = 0;
    std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;

    file.seekg(cursor);
    data_t mapped_src, mapped_dest;

    t_st = wtime();
    std::cout << "[Progress] Constructing adjacency list (Coordinate Array) . . . ";
    for(data_t i = 0; i < num_loop; i++){

        std::getline(file, line);
        if(is_verylarge)
            sscanf(line.c_str(), "%llu %llu", &v_src, &v_dest);
        else
            sscanf(line.c_str(), "%u %u", &v_src, &v_dest);
        mapped_src = mapping[v_src];
        mapped_dest = mapping[v_dest];

        adj_list[beg_pos[mapped_src] + degree[mapped_src]] = mapped_dest;

        if(is_undirected){
            if(mapped_src == mapped_dest) // self-loop
                adj_list[beg_pos[mapped_dest] + degree[mapped_dest] + 1] = mapped_src;
            else
                adj_list[beg_pos[mapped_dest] + degree[mapped_dest]] = mapped_src;
        }

        degree[mapped_src]++;
        if(is_undirected)
            degree[mapped_dest]++;
    }
    std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;
    free(mapping);
    file.close();

    if(is_sorted){

        t_st = wtime();
        std::cout << "[Progress] Sorting intra-neighbor lists . . . ";
        data_t k, temp;
        for(data_t i = 0; i < vert_count; i++){
            for(data_t j = beg_pos[i]; j < beg_pos[i + 1] - 1; j++){
                k = j;
                while(k >= beg_pos[i] && adj_list[k] > adj_list[k + 1]){
                    temp = adj_list[k];
                    adj_list[k] = adj_list[k + 1];
                    adj_list[k + 1] = temp;
                    if(k<=0) // Without this condition, underflow occurs since the data type of 'k' is 'unsigned'.
                        break;
                    else
                        k--;
                }
            }
        }
        std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;
    }

    data_t cnt_spawned = 0;
    data_t virt_vert_count;
    if(is_virtual){
        t_st = wtime();
        std::cout << "[Progress] Counting the vertices to be added virtually . . . ";
        if(max_degree == 1){
            for(data_t i = 0; i < vert_count; i++){
                if(degree[i] > max_degree)
                    cnt_spawned += (degree[i] - 1);
            }
        }
        else{
            for(data_t i = 0; i < vert_count; i++){
                if(degree[i] > max_degree)
                    cnt_spawned += (degree[i] / max_degree);
            }
        }
        std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;
        if(cnt_spawned == 0){
            is_virtual = false;
            std::cout << "[Info] There are no vertices virtually split since the degree of every vertex is not higher than " << max_degree << "." << std::endl;
            std::cout << "[Notification] --virtual disabled" << std::endl;
        }
        else{
            virt_vert_count = vert_count + cnt_spawned;
            std::cout << "[Info] " << cnt_spawned << " vertices will be virtually spawned." << std::endl;
            std::cout << "[Info] NUM_VERTICES = " << virt_vert_count << std::endl;
        }
    }

    if(!is_virtual){
        t_st = wtime();
        std::cout << "[Progress] Writing beg_pos to a file named " << filename << "_beg_pos.bin" << " . . . ";
        FILE *file_beg_pos = fopen64((filename + "_beg_pos.bin").c_str(), "wb");
        fwrite(beg_pos, vert_count + 1, sizeof(data_t), file_beg_pos);
        fclose(file_beg_pos);
        free(beg_pos);
        std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;
        disk_beg_pos = (vert_count + 1) * sizeof(data_t);
        disk_acc += disk_beg_pos;
        fin_beg_pos = true;
    }

    t_st = wtime();
    std::cout << "[Progress] Writing adj_list to a file named " << filename << "_adj_list.bin" << " . . . ";
    FILE *file_adj_list = fopen64((filename + "_adj_list.bin").c_str(), "wb");
    fwrite(adj_list, edge_count, sizeof(data_t), file_adj_list);
    fclose(file_adj_list);
    free(adj_list);
    std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;
    disk_adj_list = edge_count * sizeof(data_t);
    disk_acc += disk_adj_list;
    fin_adj_list = true;

    if(is_virtual){
        data_t *vid_pos = (data_t *) malloc(sizeof(data_t) * (vert_count + 1));
        data_t *vid_list = (data_t *) malloc(sizeof(data_t) * (virt_vert_count));
        data_t *virt_beg_pos = (data_t *) malloc(sizeof(data_t) * (virt_vert_count + 1));

        data_t split;
        vid_pos[vert_count] = virt_vert_count;
        data_t offset = 0;
        t_st = wtime();
        std::cout << "[Progress] Constructing vid_pos and vid_list . . . ";
        for(data_t i = 0; i < vert_count; i++){
            if(max_degree == 1){
                if(degree[i] > max_degree)
                    split = degree[i];
                else
                    split = 1;
            }
            else{
                if(degree[i] > max_degree)
                    split = degree[i] / max_degree + 1;
                else
                    split = 1;
            }

            vid_pos[i] = offset;
            for(data_t j = 0; j < split; j++)
                vid_list[offset + j] = i;
            offset += split;
        }
        std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;

        virt_beg_pos[0] = 0;
        virt_beg_pos[virt_vert_count] = edge_count;
        data_t cur = 1;
        data_t round = 0;
        t_st = wtime();
        std::cout << "[Progress] Constructing virtualized beginning position array . . . ";
        while(cur < virt_vert_count){
            split = vid_pos[round + 1] - vid_pos[round];
            for(data_t i = 0; i < split - 1; i++){
                virt_beg_pos[cur] = virt_beg_pos[cur - 1] + max_degree;
                cur++;
            }
            if(max_degree == 1)
                virt_beg_pos[cur] = virt_beg_pos[cur - 1] + 1;
            else{
                if(split == 1)
                    virt_beg_pos[cur] = virt_beg_pos[cur - 1] + degree[round];
                else{
                    if(degree[round] % max_degree == 0)
                        virt_beg_pos[cur] = virt_beg_pos[cur - 1] + max_degree;
                    else
                        virt_beg_pos[cur] = virt_beg_pos[cur - 1] + (degree[round] % max_degree);
                }
            }
            cur++;
            round++;
        }
        std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;

        t_st = wtime();
        std::cout << "[Progress] Writing vid_pos to a file named " << filename << "_vid_pos.bin" << " . . . ";
        FILE *file_vid_pos = fopen64((filename + "_vid_pos.bin").c_str(), "wb");
        fwrite(vid_pos, vert_count + 1, sizeof(data_t), file_vid_pos);
        fclose(file_vid_pos);
        free(vid_pos);
        std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;
        disk_vid_pos = (vert_count + 1) * sizeof(data_t);
        disk_acc += disk_vid_pos;
        fin_vid_pos = true;

        t_st = wtime();
        std::cout << "[Progress] Writing vid_list to a file named " << filename << "_vid_list.bin" << " . . . ";
        FILE *file_vid_list = fopen64((filename + "_vid_list.bin").c_str(), "wb");
        fwrite(vid_list, virt_vert_count, sizeof(data_t), file_vid_list);
        fclose(file_vid_list);
        free(vid_list);
        std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;
        disk_vid_list = virt_vert_count * sizeof(data_t);
        disk_acc += disk_vid_list;
        fin_vid_list = true;

        t_st = wtime();
        std::cout << "[Progress] Writing virt_beg_pos to a file named " << filename << "_virt_beg_pos.bin" << " . . . ";
        FILE *file_virt_beg_pos = fopen64((filename + "_virt_beg_pos.bin").c_str(), "wb");
        fwrite(virt_beg_pos, virt_vert_count + 1, sizeof(data_t), file_virt_beg_pos);
        fclose(file_virt_beg_pos);
        free(virt_beg_pos);
        std::cout << "[Done] " << "(elapsed (s): " << wtime() - t_st << ")" << std::endl;
        disk_virt_beg_pos = (virt_vert_count + 1) * sizeof(data_t);
        disk_acc += disk_virt_beg_pos;
        fin_virt_beg_pos = true;
    }

    free(degree);

    std::cout << "===============================================================" << std::endl;
    std::cout << "[Result] Successfully completed! " << std::endl;
    if(is_virtual)
        std::cout << "[Result] Number of vertices:  " << virt_vert_count << " (" << vert_count << " + " << cnt_spawned << ")" << std::endl;
    else
        std::cout << "[Result] Number of vertices:  " << vert_count << std::endl;
    std::cout << "[Result] Number of edges:  " << edge_count << std::endl;
    std::cout << "[Result] Occupied capacity (bytes):  " << disk_acc << std::endl;
    std::cout << "[Result] Consumed time (s):  " << wtime() - t_elpd_st << std::endl;
    std::cout << "===============================================================" << std::endl;
    if(fin_beg_pos)
        std::cout << "[Created] " << filename << "_beg_pos.bin" << " (" << disk_beg_pos << " bytes)" << std::endl;
    if(fin_adj_list)
        std::cout << "[Created] " << filename << "_adj_list.bin" << " (" << disk_adj_list << " bytes)" << std::endl;
    if(fin_vid_pos)
        std::cout << "[Created] " << filename << "_vid_pos.bin" << " (" << disk_vid_pos << " bytes)" << std::endl;
    if(fin_vid_list)
        std::cout << "[Created] " << filename << "_vid_list.bin" << " (" << disk_vid_list << " bytes)" << std::endl;
    if(fin_virt_beg_pos)
        std::cout << "[Created] " << filename << "_virt_beg_pos.bin" << " (" << disk_virt_beg_pos << " bytes)" << std::endl;
    std::cout << "===============================================================" << std::endl;
}

int main(int argc, char **argv){

    if(argc < 3){

        std::cout
                << "Required argument:\n"
                << "\t--input : input graph data in mtx format (e.g., --input com-Orkut.mtx)\n"
                << "Optional arguments:\n"
                << "\t--undirected : add reverse edges\n"
                << "\t--virtual : build vCSR (virtual CSR) specifying maximum degree (<= 1024) for each vertex (e.g., --virtual 32)\n"
                << "\t--sorted : sort intra-neighbor lists\n"
                << "\t--verylarge : set data type of vertices and edges to 'unsigned long long' to handle very large input graph (e.g., com-Friendster), default='unsigned int'\n"
                << std::endl;

        exit(-1);
    }

    std::string filename;
    bool is_undirected = false;
    bool is_virtual = false;
    bool is_sorted = false;
    bool is_verylarge = false;
    int max_degree;

    bool err_arg_invalid_input = false;
    bool err_arg_invalid_max_degree = false;

    bool is_checked_input = false;
    bool is_checked_undirected = false;
    bool is_checked_virtual = false;
    bool is_checked_sorted = false;
    bool is_checked_verylarge = false;

    double t_elpd_st;

    t_elpd_st = wtime();
    for(int i = 1; i < argc; i++){

        if(!strcmp(argv[i], "--input") && i != argc - 1){

            if(!is_checked_input){

                if(!strcmp(argv[i + 1] + (strlen(argv[i + 1]) - 4), ".mtx")){

                    err_arg_invalid_input = false;
                    filename = std::string(argv[i + 1]);
                }
                else
                    err_arg_invalid_input = true;

                is_checked_input = true;
            }
        }

        else if(!strcmp(argv[i], "--undirected")){

            if(!is_checked_undirected){

                is_undirected = true;
                is_checked_undirected = true;
            }
        }

        else if(!strcmp(argv[i], "--virtual") && i != argc - 1){

            if(!is_checked_virtual){

                max_degree = atoi(argv[i + 1]);

                if(max_degree > 0 && max_degree <= 1024)
                    err_arg_invalid_max_degree = false;
                else
                    err_arg_invalid_max_degree = true;

                is_virtual = true;
                is_checked_virtual = true;
            }
        }

        else if(!strcmp(argv[i], "--sorted")){

            if(!is_checked_sorted){

                is_sorted = true;
                is_checked_sorted = true;
            }
        }

        else if(!strcmp(argv[i], "--verylarge")){

            if(!is_checked_verylarge){

                is_verylarge = true;
                is_checked_verylarge = true;
            }
        }
    }

    int cnt_err = 0;

    if(err_arg_invalid_input){
        std::cout << "[Invalid] Must use mtx format (Matrix Market)." << std::endl;
        cnt_err++;
    }
    else
        std::cout << "[Valid] Input file: " << filename << std::endl;

    if(is_undirected)
        std::cout << "[Valid] Undirected (reverse edges)" << std::endl;
    else
        std::cout << "[Valid] Directed" << std::endl;

    if(is_virtual){
        if(err_arg_invalid_max_degree){
            std::cout << "[Invalid] Maximum degree must be an integer k, where 0 < k <= 1024." << std::endl;
            cnt_err++;
        }
        else
            std::cout << "[Valid] vCSR with maximum degree " << max_degree << std::endl;
    }
    else
        std::cout << "[Valid] Conventional CSR" << std::endl;

    if(is_sorted)
        std::cout << "[Valid] Sorted intra-neighbor lists" << std::endl;
    else
        std::cout << "[Valid] Unsorted intra-neighbor lists" << std::endl;

    if(is_verylarge)
        std::cout << "[Valid] Data type='unsigned long long'" << std::endl;

    else
        std::cout << "[Valid] Data type='unsigned int'" << std::endl;


    if(cnt_err > 0){
        std::cout << "\nPlease make sure that arguments are valid." << std::endl;
        exit(-1);
    }

    if(is_verylarge){

        preprocess_graph<unsigned long long>(

                filename,
                is_undirected,
                is_virtual,
                is_sorted,
                is_verylarge,
                t_elpd_st,
                max_degree,
                ULLONG_MAX
        );
    }
    else
        preprocess_graph<unsigned int>(

                filename,
                is_undirected,
                is_virtual,
                is_sorted,
                is_verylarge,
                t_elpd_st,
                max_degree,
                UINT_MAX
        );

    return 0;
}
