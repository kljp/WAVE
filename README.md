# GBRO
## GPU BFS Runtime Optimization
---
This project aims to support a high performance breadth-first graph traversal on GPUs.

---
Tested operating system
-----
Ubuntu 16.04.5 LTS, Ubuntu 20.04.1 LTS

---
Tested software
-----
g++ 5.4.0, CUDA 11.2, CUDA 11.3

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
./gbro *_beg_pos.bin *_csr.bin

---
Code specification
-----
####GBRO implementation:
- main.cu: load a graph as an input
- bfs.cuh: traverse the graph
- fqg.cuh: implementation of traversals of top-down and bottom-up
- mcpy.cuh: functions for initializing data structures
- alloc.cuh: memory allocation for data structures
- comm.cuh: global variables and functions shared by all files

####Headers Provided by https://github.com/iHeartGraph/Enterprise/:
- graph.h: graph data structure
- graph.hpp: implementation of graph data structure
- wtime.h: get current time for measuring the consumed time

####CSR Generator provided by https://github.com/hpda-lab/XBFS/:
- csrg.cpp: generate CSR
    - Compile: make
    - Execute: ./csrg <reverse> <lines_to_skip>
    - Graph source: use one in your preference
        - Text file (.txt): https://snap.stanford.edu/data/
        - Matrix market (.mtx): https://sparse.tamu.edu/SNAP/