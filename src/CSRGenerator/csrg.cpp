/*
 * Copyright 2016 The George Washington University
 * Written by Hang Liu 
 * Directed by Prof. Howie Huang
 *
 * https://www.seas.gwu.edu/~howie/
 * Contact: iheartgraph@gmail.com
 *
 * 
 * Please cite the following paper:
 * 
 * Hang Liu and H. Howie Huang. 2015. Enterprise: breadth-first graph traversal on GPUs. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '15). ACM, New York, NY, USA, Article 68 , 12 pages. DOI: http://dx.doi.org/10.1145/2807591.2807594
 
 *
 * This file is part of Enterprise.
 *
 * Enterprise is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Enterprise is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Enterprise.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include "../wtime.h"

#define SIZE_LIMIT (1<<30)

#define INFTY int(1<<30)
using namespace std;

typedef long int vertex_t;
typedef long int index_t;

inline off_t fsize(const char *filename) {
    struct stat st;
    if (stat(filename, &st) == 0)
        return st.st_size;
    return -1;
}


int main(int argc, char** argv){
    int fd,i;
    char* ss_head;
    char* ss;

    std::cout<<"Input: ./exe tuple_file(text) "
             <<"reverse_the_edge(1 reverse, 0 not reverse) lines_to_skip\n";
    if(argc<4){printf("Wrong input\n");exit(-1);}


    size_t file_size = fsize(argv[1]);
    bool is_reverse=(atol(argv[2])==1);
    long skip_head=atol(argv[3]);

    fd=open(argv[1],O_CREAT|O_RDWR,00666 );
    if(fd == -1)
    {
        printf("%s open error\n", argv[1]);
        perror("open");
        exit(-1);
    }

    ss_head = (char*)mmap(NULL,file_size,PROT_READ|PROT_WRITE,MAP_PRIVATE,fd,0);
    assert(ss_head != MAP_FAILED);
    madvise(ss_head, file_size, MADV_SEQUENTIAL);
    double time_beg = wtime();
    long progress = 1;

    size_t head_offset=0;
    int skip_count = 0;
    while(true)
    {
        if(skip_count == skip_head) break;
        if(head_offset == file_size &&
           skip_count < skip_head)
        {
            std::cout<<"Eorr: skip more lines than the file has\n\n\n";
            exit(-1);
        }

        head_offset++;
        if(ss_head[head_offset]=='\n')
        {
            head_offset++;
            skip_count++;
            if(skip_count == skip_head) break;
        }
        if(head_offset > progress)
        {
            printf("%ld lines processed, %f seconds elapsed\n", head_offset, wtime() - time_beg);
            progress <<=1;
        }
    }

    ss = &ss_head[head_offset];
    file_size -= head_offset;

    size_t curr=0;
    size_t next=0;

    //step 1. vert_count,edge_count,
    size_t edge_count=0;
    size_t vert_count;
    vertex_t v_max = 0;
    vertex_t v_min = INFTY;//as infinity
    vertex_t a;

    progress = 1;
    while(next<file_size){
        char* sss=ss+curr;
        a = atol(sss);

        if(v_max<a){
            v_max = a;
        }
        if(v_min>a){
            v_min = a;
        }

        while((ss[next]!=' ')&&(ss[next]!='\n')&&(ss[next]!='\t')){
            next++;
        }
        while((ss[next]==' ')||(ss[next]=='\n')||(ss[next]=='\t')||(ss[next]=='\r')){
            next++;
        }
        curr = next;
        if(next > progress)
        {
            printf("%f%%, %f seconds elapsed\n", next*100.0/file_size, wtime() - time_beg);
            progress <<=1;
        }

        //one vertex is counted once
        edge_count++;
    }

    const index_t line_count=edge_count>>1;
    if(!is_reverse) edge_count >>=1;

    vert_count = v_max + 1;
    assert(v_min<INFTY);
    cout<<"edge count: "<<edge_count<<endl;
    cout<<"max vertex id: "<<v_max<<endl;
    cout<<"min vertex id: "<<v_min<<endl;

    cout<<"edge count: "<<edge_count<<endl;
    cout<<"vert count: "<<vert_count<<endl;

    //step 2. each file size
    char filename[256];
    sprintf(filename,"%s_csr.bin",argv[1]);
    FILE *fd4 = fopen64(filename, "wb");
    assert(fd4 != NULL);

    vertex_t* adj = (vertex_t*)mmap(NULL,
                                    edge_count*sizeof(vertex_t),
                                    PROT_READ|PROT_WRITE,
                                    MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    assert(adj != MAP_FAILED);

    index_t *degree = (index_t *)mmap (NULL,
                                       vert_count*sizeof(index_t),
                                       PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS,
                                       -1,
                                       0);
    assert(degree != MAP_FAILED);

    sprintf(filename,"%s_beg_pos.bin",argv[1]);
    int fd3 = open(filename,O_CREAT|O_RDWR,00666 );
    ftruncate(fd3, (vert_count+1)*sizeof(index_t));
    index_t* begin  = (index_t*)mmap(NULL,(vert_count+1)*sizeof(index_t),PROT_READ|PROT_WRITE,MAP_SHARED,fd3,0);
    assert(begin != MAP_FAILED);

    //step 3. write degree
    for(int i=0; i<vert_count;i++){
        degree[i]=0;
    }

    vertex_t index, dest;
    size_t offset =0;
    curr=0;
    next=0;

    printf("Getting degree progress ...\n");
    progress = 1;

    while(offset<line_count){
        char* sss=ss+curr;
        index = atol(sss);
        while((ss[next]!=' ')&&(ss[next]!='\n')&&(ss[next]!='\t')){
            next++;
        }
        while((ss[next]==' ')||(ss[next]=='\n')||(ss[next]=='\t')){
            next++;
        }
        curr = next;

        char* sss1=ss+curr;
        dest=atol(sss1);

        while((ss[next]!=' ')&&(ss[next]!='\n')&&(ss[next]!='\t')){
            next++;
        }
        while((ss[next]==' ')||(ss[next]=='\n')||(ss[next]=='\t')||(ss[next]=='\r')){
            next++;
        }
        curr = next;
        degree[index]++;
        if(is_reverse) degree[dest]++;
        if(offset > progress)
        {
            printf("%f%%, %f seconds elapsed\n", offset*100.0/line_count, wtime() - time_beg);
            progress <<=1;
        }

        offset++;
    }

    begin[0]=0;
    begin[vert_count]=edge_count;

    printf("\nCalculate beg_pos ...\n");
    for(size_t i=1; i<vert_count; i++){
        begin[i] = begin[i-1] + degree[i-1];
        degree [i-1] = 0;
    }
    degree[vert_count-1] = 0;
    //step 4: write adjacent list
    vertex_t v_id;
    offset =0;
    next = 0;
    curr = 0;

    progress = 1;
    printf("\nConstructing CSR progress...\n");
    while(offset<line_count){
        char* sss=ss+curr;
        index = atol(sss);
        while((ss[next]!=' ')&&(ss[next]!='\n')&&(ss[next]!='\t')){
            next++;
        }
        while((ss[next]==' ')||(ss[next]=='\n')||(ss[next]=='\t')||(ss[next]=='\r')){
            next++;
        }
        curr = next;

        char* sss1=ss+curr;
        v_id = atol(sss1);
        adj[begin[index]+degree[index]] = v_id;

        if(is_reverse){
            if(v_id==index) //self-loop
                adj[begin[v_id]+degree[v_id] + 1] = index;
            else // normal case
                adj[begin[v_id]+degree[v_id]] = index;
        }

        while((ss[next]!=' ')&&(ss[next]!='\n')&&(ss[next]!='\t')){
            next++;
        }
        while((ss[next]==' ')||(ss[next]=='\n')||(ss[next]=='\t')||(ss[next]=='\r')){
            next++;
        }
        curr = next;
        degree[index]++;
        if(is_reverse) degree[v_id]++;
        if(offset > progress)
        {
            printf("%f%%, %f seconds elapsed\n", offset*100.0/line_count, wtime() - time_beg);
            progress <<=1;
        }

        offset++;
    }


    long long int to_write_size = edge_count*sizeof(vertex_t);
    long long int to_write_off = 0;
    std::cout<<"Dumping "<<to_write_size<<" bytes CSR to disk ...\n";

    std::cout<<"Adj Write "<<fwrite(adj, edge_count, sizeof(vertex_t), fd4)<<" bytes and expect "<< edge_count*sizeof(vertex_t)<<" bytes\n";

    printf("%f seconds elapsed\n\n", wtime() - time_beg);

    munmap( ss,sizeof(char)*file_size );

    munmap( adj,sizeof(vertex_t)*edge_count );
    munmap( begin,sizeof(index_t)*vert_count+1 );
    munmap( degree,sizeof(index_t)*vert_count );

    close(fd3);
    fclose(fd4);
}