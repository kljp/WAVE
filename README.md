# GBRO
## GPU BFS Runtime Optimization
---
This project aims to support a high performance breadth-first graph traversal on GPUs.

---
Tested software
-----
g++ 5.4.0, CUDA 11.2, CUDA 11.3

---
Tested hardware
-----
GTX970, RTX3080 (more products will be tested)

---
Compile
-----
make

---
Execute
-----
./gbro *_beg_pos.bin *_csr.bin

---
Converter: edge tuples to CSR
----
- Compile: make
- Execute: type "./text_to_bin.bin", it will show you what is needed
- Basically, you could download a tuple list file from [snap](https://snap.stanford.edu/data/). Afterwards, you could use this converter to convert the edge list into CSR format. 

**For example**:

- Download https://snap.stanford.edu/data/com-Orkut.html file. **unzip** it. 
- **./text_to_bin.bin soc-orkut.mtx 1 2(could change, depends on the number of lines are not edges)**
- You will get *soc-orkut.mtx_beg_pos.bin* and *soc-orkut.mtx_csr.bin*. 
- You could use these two files to run GBRO.

---
Code specification
-----
TBD
