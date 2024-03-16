#include <iostream>
#include "Value.h"

using namespace dag;

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

    graph.PrintGraph();

    *graph[graph.size() - 1] = Node(3);
    graph.PrintGraph();

    return 0;
}
