#include "Value.h"

using namespace DAG;

int main() {
    Graph graph;

    Value rand1(graph);
    Value rand2(graph);
    Value five(5, graph);

    rand1.SetRandom();
    rand2.SetRandom();

    auto rand3 = rand1 + rand2;
    auto rand4 = five * rand3;
    
    graph.PrintGraph();

    graph.clear();

    rand1.SetRandom();
    rand2.SetRandom();

    rand4 = rand1 / rand2;

    graph.PrintGraph();

    return 0;
}
