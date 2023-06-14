all: compile
compile:
		nvcc -std=c++11 src/main.cu src/gpu/bfs_gpu_opt.cu src/gpu/bfs_gpu.cu src/gpu/bfs_gpu1.cu src/graph/graph.cpp src/cpu/bfs_cpu.cpp -o Parallel-BFS
