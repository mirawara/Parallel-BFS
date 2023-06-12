#ifndef BFS_GPU_CUH
#define BFS_GPU_CUH

#include <bits/stdc++.h>

#include "../../graph/graph.h"

/*
 * start - vertex number from which traversing a graph starts
 * G - graph to traverse
 * distance - placeholder for vector of distances (filled after invoking a function)
 * visited - placeholder for vector indicating that vertex was visited (filled after invoking a function)
 */
void bfsGPU(int start, Graph &G, std::vector<int> &distance, std::vector<bool> &visited, std::ofstream *file, int block_size);

void bfsGPUOpt(int start, Graph& G, std::vector<int>& distance, std::vector<bool>& visited, std::ofstream* file);

void bfsGPUMethod1Opt(int start, Graph &G, std::vector<int> &distance, std::vector<bool> &visited, std::ofstream *file, int block_size);

#endif // BFS_GPU_CUH
