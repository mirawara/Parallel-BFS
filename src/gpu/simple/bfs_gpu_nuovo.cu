#include "bfs_gpu.cuh"

using namespace std;

#define DEBUG(x)
#define N_THREADS_PER_BLOCK 32
#define N 10

__global__
void computeNextQueue1(int *adjacencyList, int *edgesOffset, int *edgesSize, int *distance,
                      int queueSize, int *currentQueue, int *nextQueueSize, int *nextQueue, int level) {
    /* Thread ID */
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    /* Check */
    if (tid*N < queueSize) {

        /* Every node in the queue is assigned to a specidic thread */
        int current;

        int n = queueSize - (tid * N);
        if (n >= N) n = N;

       #pragma unroll
        for (int j = 0; j < n; j++) {
            /* Check for unvisited neighbors */
            current = currentQueue[tid*N+j];

            #pragma unroll
            for (int i = edgesOffset[current]; i < edgesOffset[current] + edgesSize[current]; ++i) {

                int v = adjacencyList[i];

                if (distance[v] == INT_MAX) { /* => unvisited */

                    /* Mark with the distance from the starting point */
                    distance[v] = level + 1;

                    /* Increment the shared variable (necessary) */
                    int position = atomicAdd(nextQueueSize, 1);
                   // printf("nextqueuesize %d/n", *nextQueueSize);

                    /* Add the node in the next queue */
                    nextQueue[position] = v;
                }
                
            }
        }
    }
    
}


void bfsGPUOpt(int start, Graph &G, vector<int> &distance, vector<bool> &visited, std::ofstream *file) {

    

   //const int n_blocks = (G.numVertices + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

    int n_blocks;

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

    /* Start measuring execution time */
    cudaEvent_t startEv, stopEv;
    cudaEventCreate(&startEv);
    cudaEventCreate(&stopEv);
    cudaEventRecord(startEv);

    /* Initiale and copy the output */
    distance = vector<int>(G.numVertices, INT_MAX);
    distance[start] = 0;
    cudaMemcpy(d_distance, distance.data(), size, cudaMemcpyHostToDevice);

    int* d_currentQueue;
    int* d_nextQueue;

    d_currentQueue = d_firstQueue;
    d_nextQueue = d_secondQueue;

    /* While there are node to visit */
    while (currentQueueSize > 0) {

        int a = ceil(currentQueueSize / N) + 1;
        
        n_blocks = (a + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

        /* Calling the kernel with block of 32x32 threads, at least #vertices threads in total */
        computeNextQueue1<<<n_blocks, N_THREADS_PER_BLOCK>>>(d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
                                                            currentQueueSize, d_currentQueue, d_nextQueueSize,
                                                            d_nextQueue, level);

        /* Increment of the level */
        ++level;
       // printf("level %d\n", level);
        /* The next queue of the previous step becomes the current queue */
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

    /* Retrieve the output */
    cudaMemcpy(&distance[0], d_distance, size, cudaMemcpyDeviceToHost);

    /* Measuring performance */
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
    *file << elapsed << ";";
}
