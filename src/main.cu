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

    vertex_t src = rand() % ginst->vert_count;

    bfs<vertex_t, index_t>(

        src,
        ginst->beg_pos,
        ginst->csr,
        ginst->vert_count,
        ginst->edge_count,
        gpu_id
    );

    delete ginst;

    return 0;
}