#include "graph.h"
#include "bfs.cuh"
#include <sstream>
#include <fstream>

using namespace std;

int main(int args, char **argv){

    typedef int vertex_t;
    typedef int index_t;

    if(args != 3){
        
        cout << "Wrong input" << endl;
        return -1;
    }
    
    const vertex_t gpu_id = 0;
    graph<long, long, double, vertex_t, index_t, double> *ginst
    = new graph<long, long, double, vertex_t, index_t, double>(argv[1], argv[2], NULL);

    vertex_t *src_list = new int[NUM_ITER];
    vertex_t src;
    for(index_t i = 0; i < NUM_ITER; i++){

        src = rand() % ginst->vert_count;

        if(ginst->beg_pos[src + 1] - ginst->beg_pos[src] > 0)
            src_list[i] = src;
        else
            i--;
    }

    bfs<vertex_t, index_t>(

        src_list,
        ginst->beg_pos,
        ginst->csr,
        ginst->vert_count,
        ginst->edge_count,
        gpu_id
    );

    delete[] src_list;
    delete ginst;

    return 0;
}