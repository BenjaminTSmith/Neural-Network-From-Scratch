#include <iostream>
#include "Value.h"

using namespace dag;

int main() {

    Graph graph;
    Value two(2);
    Value three(3);
    Value seven(7);
    Value thirteen(13);
    Value nine(9);

    two.SetGraph(graph);
    three.SetGraph(graph);
    seven.SetGraph(graph);
    thirteen.SetGraph(graph);
    nine.SetGraph(graph);

    auto twelve = two + three + seven;
    auto twentyfive = thirteen + twelve;
    auto seventyfive = three * twentyfive;

    graph.PrintGraph();
    
    return 0;
}
