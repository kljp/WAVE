# GBRO
## GPU BFS Runtime Optimization
---
This project aims to support a high performance breadth-first graph traversal on GPUs.

---
Tested operating system
-----
Ubuntu \[16.04.5, 18.04.5, 20.04.2\] LTS

---
Tested software
-----
g++ \[5.4.0, 7.5.0, 9.3.0\], CUDA \[11.2, 11.3, 11.4\]

---
Tested hardware
-----
GTX970, RTX3080

---
Compile
-----
make

---
Execute
-----
./gbro \<\*_beg_pos.bin\> \<\*_adj_list.bin\>

---
Code specification
-----
__GBRO implementation:__
- main.cu: load a graph as an input
- bfs.cuh: traverse the graph
- fqg.cuh: implementation of traversals of top-down and bottom-up
- mcpy.cuh: functions for initializing data structures
- alloc.cuh: memory allocation for data structures
- comm.cuh: global variables and functions shared by all files

__CSR Generator provided by https://github.com/kljp/vCSR/:__
- vcsr.cpp: generate CSR
    - Compile: make
    - Execute: ./vcsr --input \<\*.mtx\> \[option1\] \[option2\] \[option3\] \[option4\]
      - \[option1\]: --virtual \<max\_degree\> \(not available for GBRO\)
        - set maximum degree of a vertex to \<max\_degree\>
      - \[option2\]: --undirected
        - add reverse edges
      - \[option3\]: --sorted
        - sort intra-neighbor lists
      - \[option4\]: --verylarge
        - set data type of vertices and edges to 'unsigned long long', default='unsigned int'
    - Graph source: https://sparse.tamu.edu/
    - Please make sure that the format of input graph should be Matrix Market.

__Headers Provided by https://github.com/iHeartGraph/Enterprise/:__
- graph.h: graph data structure
- graph.hpp: implementation of graph data structure
- wtime.h: get current time for measuring the consumed time
---
Contact
-----
If you have any questions about this project, contact me by one of the followings:
- slashxp@naver.com
- kljp@ajou.ac.kr