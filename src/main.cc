#include "Value.h"

using namespace DAG;

int main() {
    Graph graph;

    Value two(2, graph);
    Value three(3, graph);
    Value seven(7, graph);
    Value nine(9, graph);
    Value thirteen(13, graph);
    Value negative_nine(-9, graph);

    auto one = three - two;
    auto sixteen = seven + nine;
    auto sixtythree = seven * nine;
    auto zero = negative_nine.ReLU();
    auto pos_seven = seven.ReLU();

    graph.PrintGraph();

    graph.clear();

    sixteen = two * three;
    graph.PrintGraph();

    return 0;
}
