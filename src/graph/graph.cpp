#include "graph.h"
#include <iostream>
#include <fstream>


using namespace std;


void print_vector(string title, vector<int> &v);

//Builds the adjacency list as a vector of vectors, each one of them containing the neighbors of a node
Graph::Graph(Direction direction) {

    std::ifstream file("graphs/128000_100.graph", std::ifstream::in);

    int numVertices, numEdges;
    cout << "Started reading graph" << endl;
    (istream&)file >> numVertices >> numEdges;
   // printf("Bu");
    vector <vector<int>> adjacencyList(numVertices);
    string line;
    for (int v = 0; v < numVertices; ++v) {
        //printf("Poi te ne restano mille\n");
        getline((istream&)file, line);
        //("Sa sa sa sabato sera\n");
        stringstream splitter(line);
        int w;
        while (splitter >> w) {
            //printf("%d\n", w);
            adjacencyList[v].push_back(w);
            if (direction == Undirected) {
                adjacencyList[w].push_back(v);
          //      printf("hehe\n");
            }
        }
    }
    file.close();
    this->init(adjacencyList, numEdges);
    cout << "Finished reading graph" << endl;

}

//Initializes the structure of the Graph object
void Graph::init(vector <vector<int>> adjacencyList, int numEdges) {
    const int numVertices = adjacencyList.size();

    // Creation of single vector adjacency list
    for (int i = 0; i < numVertices; ++i) {
        this->edgesOffset.push_back(this->adjacencyList.size());
        this->edgesSize.push_back(adjacencyList[i].size());
        for (int v: adjacencyList[i]) {
            this->adjacencyList.push_back(v);
        }
    }
    this->numVertices = numVertices;
    this->numEdges = numEdges;
}


void Graph::print() {
    printf("Graph(numVertices = %i, numEdges = %i)\n", numVertices, numEdges);
    print_vector("AdjacencyList:", adjacencyList);
    print_vector("edgesOffset:", edgesOffset);
    print_vector("edgesSize:", edgesSize);
}


void print_vector(string title, vector<int> &v) {
    cout << title << " { ";
    for (int i = 0; i < v.size(); ++i) {
        cout << v[i];
        if (i < v.size() - 1)
            cout << ", ";
    }
    cout << " }" << endl;
}
