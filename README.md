<h1 align="center" id="title">Parallel BFS</h1>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-C-76B900.svg?style=flat&logo=nvidia&logoColor=white" alt="CUDA C">
    <img src="https://img.shields.io/badge/NVIDIA-GPU-76B900.svg?style=flat&logo=nvidia&logoColor=white" alt="NVIDIA">

</p>


<p id="description">This project is a parallel implementation of the Breadth-First Search (BFS) algorithm for graph traversal using CUDA programming on NVIDIA GPUs. The goal is to improve the performance of BFS on large-scale graphs by exploiting the parallelism of GPUs. The project includes a detailed analysis of the algorithm, implementation, kernel profile, and performance evaluation. The implementation is optimized using various techniques such as memory coalescing, thread work distribution, and shared memory usage. The kernel profile analysis provides insights into the performance bottlenecks and helps in optimizing the implementation further. The performance evaluation is done on various graphs of different sizes and densities, and the results show significant speedup compared to the sequential BFS algorithm on CPU. See the <a href="https://github.com/mirawara/Parallel-BFS/Wiki/BFS-Wiki">Wiki</a> for further details.
</p>


<h2>🛠️ Tools for Performance evaluation:</h2>

* <a href="https://developer.nvidia.com/nsight-systems">Nvidia Nsight Systems</a>
* <a href="https://developer.nvidia.com/nsight-compute">Nvidia Nsight Compute</a>

<h2>🌳 Content tree:</h2>

```
.
├── graphs
│   ├── 16000_50.graph
│   ├── 24000_75.graph
│   ├── 32000_100.graph
│   ├── 40000_125.graph
│   ├── 48000_150.graph
│   ├── 56000_175.graph
│   ├── 64000_200.graph
│   ├── 8000_25.graph
│   └── graph_generation_depth.py
├── Makefile
├── Parallel-BFS
├── README.md
├── src
│   ├── cpu
│   │   ├── bfs_cpu.cpp
│   │   └── bfs_cpu.h
│   ├── gpu
│   │   ├── bfs_gpu1.cu
│   │   ├── bfs_gpu.cu
│   │   ├── bfs_gpu.cuh
│   │   └── bfs_gpu_opt.cu
│   ├── graph
│   │   ├── graph.cpp
│   │   └── graph.h
│   ├── main.cpp
│   └── main.cu
└── src_threads
    ├── bfs_pth_only.cpp
    └── generateGraph.cpp
```
- `graphs/`: Contains sample graphs and the Python script to generate them.
- `src/cpu/`: CPU implementation.
- `src/gpu/`: GPU implementations.
- `src_threads/`: Parallel CPU implementation using threads.
- `main.cpp`: Contains the main function for the CPU implementation.
- `main.cu`: Contains the main function for the GPU implementations.

The execution of the `main.cu` file takes a long time because the algorithm is run 30 times to obtain meaningful results (see the <a href="https://github.com/mirawara/Parallel-BFS/Wiki/BFS-Wiki">Wiki</a> for further details). You can easily modify it to suit your needs.

<h2>🖥️ Usage: </h2>
<p>Manual:</p>


```
./Parallel-BFS < graph
 ```
 
 <p>Example: </p>
 
 ```
./Parallel-BFS < graphs/64000_200.graph
 ```

The results are then stored in 'results.csv'.
<h2>💖Like my work?</h2>

Contact me if you have any corrections or additional features to offer.

<h2>👥 Authors:</h2>
<ul>
  <li><a href="https://github.com/mirawara">Lorenzo Mirabella</a></li>
  <li><a href="https://https://github.com/EdoardoLoni">Edoardo Loni</a></li>


  Some code was taken from this <a href="https://github.com/kaletap/bfs-cuda-gpu">repository</a>
</ul>


