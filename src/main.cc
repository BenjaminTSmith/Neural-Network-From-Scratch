#include "Value.h"

using namespace DAG;

int main() {
    Graph graph;

    Value x1(2.0, graph);
    Value x2(0.0, graph);

    Value w1(-3.0, graph);
    Value w2(1.0, graph);

    Value b(6.8813735870195432, graph);

    Value x1w1x2w2 = x1 * w1 + x2 * w2;
    Value o = x1w1x2w2 + b;
    // Value o = n.ReLU();

    graph.TopologicalSort();
    graph.BackProp();

    graph.PrintGraph();

    return 0;
}
