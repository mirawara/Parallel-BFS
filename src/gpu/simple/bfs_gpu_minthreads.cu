#include "bfs_gpu.cuh"

using namespace std;

#define DEBUG(x)
#define N_THREADS_PER_BLOCK 32


__global__
void computeNextQueueOpt(int *adjacencyList, int *edgesOffset, int *edgesSize, int *distance,
                         int queueSize, int *currentQueue, int *nextQueueSize, int *nextQueue, int level) {

    /* Id used to locate a thread in the grid */
    const int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    /* Check */
    if (node_id >= queueSize)
        return;

    /* Each column works on a specific node */
    int current = currentQueue[node_id];

    /* Number of neighbors */
    int i = edgesOffset[current];

    if (col < edgesSize[current]) {

        /* Every thread on a column visit a neighbor */
        int v = adjacencyList[i + col];

        if (distance[v] == INT_MAX) {

            distance[v] = level + 1;
            int position = atomicAdd(nextQueueSize, 1);
            nextQueue[position] = v;

        }
        
    }
    

}


void bfsGPUMethod1Opt(int start, Graph &G, vector<int> &distance, vector<bool> &visited, std::ofstream *file, int block_size) {

    /* Max number of neighbors */
    int max = *max_element(G.edgesSize.begin(), G.edgesSize.end());

    /* Initialization of GPU variables */
    int *d_adjacencyList;
    int *d_edgesOffset;
    int *d_edgesSize;
    int *d_firstQueue;
    int *d_secondQueue;
    int *d_nextQueueSize;
    int *d_distance; // output


    /* Initialization of CPU variables */
    int currentQueueSize = 1;
    const int NEXT_QUEUE_SIZE = 0;
    int level = 0;

    /* Allocation on device */
    const int size = G.numVertices * sizeof(int);
    const int adjacencySize = G.adjacencyList.size() * sizeof(int);
    cudaMalloc((void **) &d_adjacencyList, adjacencySize);
    cudaMalloc((void **) &d_edgesOffset, size);
    cudaMalloc((void **) &d_edgesSize, size);
    cudaMalloc((void **) &d_firstQueue, size);
    cudaMalloc((void **) &d_secondQueue, size);
    cudaMalloc((void **) &d_distance, size);
    cudaMalloc((void **) &d_nextQueueSize, sizeof(int));


    /* Copy to device */
    cudaMemcpy(d_adjacencyList, &G.adjacencyList[0], adjacencySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgesOffset, &G.edgesOffset[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgesSize, &G.edgesSize[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nextQueueSize, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_firstQueue, &start, sizeof(int), cudaMemcpyHostToDevice);

    /* Record performance */
    cudaEvent_t startEv, stopEv;
    cudaEventCreate(&startEv);
    cudaEventCreate(&stopEv);
    cudaEventRecord(startEv);

    /* Initialize and copy output */
    distance = vector<int>(G.numVertices, INT_MAX);
    distance[start] = 0;
    cudaMemcpy(d_distance, distance.data(), size, cudaMemcpyHostToDevice);

    //int block_size = 10;

    /* Compute the optimized resources
     * --> Explained below */
    int y_grid = 1;
    int y_dim;

    if (max <= block_size)
        y_dim = max;
    else {
        y_dim = block_size;
        y_grid = ceil((double) max / (double) block_size);
    }

    int* d_currentQueue;
    int* d_nextQueue;

    d_currentQueue = d_firstQueue;
    d_nextQueue = d_secondQueue;

    while (currentQueueSize > 0) {

        /* Allocate the exact number of resources used =>
         * A column for each node to visit and
         * a row for the maximum number of neighbors */
        int x_grid = 1;
        int x_dim;
        if (currentQueueSize <= block_size)
            x_dim = currentQueueSize;
        else {
            x_dim = block_size;
            x_grid = ceil((double) currentQueueSize / (double) block_size);
        }
        dim3 grid(x_grid, y_grid);
        dim3 block(x_dim, y_dim);

        /* Call the kernel with the optimized resources */
        computeNextQueueOpt<<<grid, block>>>(d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
                                             currentQueueSize, d_currentQueue, d_nextQueueSize,
                                             d_nextQueue, level);

        /* Increment of the level*/
        level++;

        if (level % 2 == 0) {
            d_currentQueue = d_firstQueue;
            d_nextQueue = d_secondQueue;
        }
        else {
            d_currentQueue = d_secondQueue;
            d_nextQueue = d_firstQueue;
        }

        cudaDeviceSynchronize();

        /* New queues size */
        cudaMemcpy(&currentQueueSize, d_nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_nextQueueSize, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
    }

    /* Copy the output */
    cudaMemcpy(&distance[0], d_distance, size, cudaMemcpyDeviceToHost);

    /* Record performance */
    cudaEventRecord(stopEv);
    cudaDeviceSynchronize();
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, startEv, stopEv);

    /* Cleanup */
    cudaFree(d_adjacencyList);
    cudaFree(d_edgesOffset);
    cudaFree(d_edgesSize);
    cudaFree(d_firstQueue);
    cudaFree(d_secondQueue);
    cudaFree(d_distance);

    /* Write performance on file */
    *file << elapsed << ";" << endl;
}
