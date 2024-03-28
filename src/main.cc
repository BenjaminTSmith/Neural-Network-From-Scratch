#include "Value.h"

using namespace DAG;

int main() {
    Graph graph;

    Value x1(2.0, graph);
    Value x2(0.0, graph);

    Value w1(-3.0, graph);
    Value w2(1.0, graph);

    Value b(6.8813735870195432, graph);

    Value x1w1 = x1 * w1;
    Value x2w2 = x2 * w2;
    Value x1w1x2w2 = x1w1 + x2w2;
    Value n = x1w1x2w2 + b;
    
    graph.TopologicalSort();
    graph.BackProp();

    graph.PrintGraph();

    for (const auto& node : graph.nodes()) {
        std::cout << node->value_ << " : " << node->grad_ << std::endl;
    }

    return 0;
}
