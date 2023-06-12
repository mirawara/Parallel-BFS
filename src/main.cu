#include <iostream>
#include <chrono>

#include "graph/graph.h"
#include "cpu/bfs_cpu.h"
#include "gpu/simple/bfs_gpu.cuh"
#include "gpu/optimized/bfs_gpu_opt.cuh"

using namespace std;


void print(vector<int> &v);

//Checks the result of the execution of a GPU implementation, comparing it with the one produced by the sequential algorithm
class Checker {
    vector<int> expected_answer;
public:

    //Vector containing the correct result (node distances computed by the CPU execution)
    Checker(vector<int> exp_ans) : expected_answer(exp_ans) {}

    //Counts the number of nodes that have been visited, and retrieves the computed depth
    pair<int, int> count_visited_vertices(const vector<int> &distance) {
        int depth = 0;
        int count = 0;
        for (int x: distance) {
            if (x < INT_MAX) {
                ++count;
                if (x > depth) {
                    depth = x;
                }
            }
        }
        return {count, depth};
    }

    //Checks that the distance vector computed by a GPU execution is equal to the one obtained by the sequential execution
    void check(vector<int> answer) {
        assert(answer.size() == expected_answer.size());
        bool is_ok = true;
        int n_errors = 0;
        int position_wrong = -1;
        for (int i = 0; i < answer.size(); ++i) {
            if (answer.at(i) != expected_answer.at(i)) {
                is_ok = false;
                position_wrong = i;
                n_errors++;
                //printf("Answer at %i equals %i but should be equal to %i\n", position_wrong, answer[position_wrong],
                      // expected_answer[position_wrong]);
            }
        }
        if (is_ok) {
            pair<int, int> graph_output = count_visited_vertices(answer);
        } else {
            printf("Something went wrong!\n");
            printf("Errors %d\n", n_errors);
        }
    }
};


// Tests speed of a BFS algorithm
int main() {

    //Creates a new Graph object
    Graph G(Undirected);
    int startVertex;
    vector<int> distance;
    vector<bool> visited;

    startVertex = 0;

    //In this file the execution times for each implementation of the algorithm will be stored
    std::ofstream file("results.csv");

    //30 repetitions in order to get statistically meaningful results
    for (int i = 0; i < 30; i++) {

        // CPU sequential algorithm
        distance = vector<int>(G.numVertices);
        visited = vector<bool>(G.numVertices);

        clock_t startEv = clock();
        bfsCPU(startVertex, G, distance, visited);
        clock_t endEv = clock();

        double elapsedTime = (double) (endEv - startEv) * 1000 / CLOCKS_PER_SEC;
        printf("Elapsed time for CPU implementation : %f ms.\n", elapsedTime);
        file << elapsedTime << ";";

        Checker checker(distance);

        // GPU Method 1 execution


        
            distance = vector<int>(G.numVertices);

            bfsGPU(startVertex, G, distance, visited, &file, 32);
            checker.check(distance);
       

        distance = vector<int>(G.numVertices);
        
        bfsGPUOpt(startVertex, G, distance, visited, &file);
        checker.check(distance);
        /*
        // GPU Method 2 execution
        distance = vector<int>(G.numVertices);

        bfsGPUMethod2(startVertex, G, distance, visited, &file);
        checker.check(distance);
        */
        // GPU Method 1 Opt execution

      //  for (int i = 1; i <= 32; i++) {

            distance = vector<int>(G.numVertices);

            bfsGPUMethod1Opt(startVertex, G, distance, visited, &file, 12);
            checker.check(distance);
       // }
        /*
        // GPU Method 2 Opt execution
        distance = vector<int>(G.numVertices);

        bfsGPUMethod2Opt(startVertex, G, distance, visited, &file);
        checker.check(distance);*/
    }
    file.close();

    return 0;
}


void print(vector<int> &v) {
    cout << "{ ";
    for (int i = 0; i < v.size(); ++i) {
        cout << v[i];
        if (i < v.size() - 1)
            cout << ", ";
    }
    cout << " }" << endl;
}
 
