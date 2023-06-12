all: compile
compile:
		nvcc -std=c++11 src/main.cu src/gpu/optimized/bfs_gpu_opt_BEST.cu src/gpu/optimized/bfs_gpu_opt.cu src/gpu/simple/bfs_gpu_minthreads.cu src/gpu/simple/bfs_gpu.cu src/graph/graph.cpp src/cpu/bfs_cpu.cpp -o ParallelBFS
