#include <iostream>
#include "Value.h"

using namespace DAG;

int main() {

    Graph graph;

    Value nineteen(19, graph);
    Value two(2, graph);
    Value three(3, graph);
    Value seven(7, graph);
    Value thirteen(13, graph);
    Value nine(9, graph);

    auto fourtynine = three + seven + thirteen * three;
    auto fiftyone = fourtynine + two;

    fiftyone.node_->grad_ = 1;
    graph.BackProp();
    std::cout << three.node_->grad_ << std::endl;

    graph.PrintGraph();

    for (auto& node : graph.nodes())
        node->value_ -= node->grad_;

    graph.PrintGraph();

    return 0;
}
