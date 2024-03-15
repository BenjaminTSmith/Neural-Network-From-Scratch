#include <iostream>
#include "Value.h"

using namespace dag;

int main() {

    Graph graph;
    Value two(2);
    Value three(3);
    Value seven(7);

    two.SetGraph(graph);
    three.SetGraph(graph);
    seven.SetGraph(graph);

    Value twelve = two + three + seven;

    graph.PrintGraph();
    
    return 0;
}
