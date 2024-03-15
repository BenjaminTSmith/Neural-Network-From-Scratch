#ifndef GRAPH_H
#define GRAPH_H

#include "Node.h"

namespace dag {

class Graph {
public:
    std::vector<Node> nodes_;
    
    Graph() {}

    Graph(size_t size) : nodes_(size) {}

    void AddNode(const Node& node) {
        nodes_.push_back(Node(node));
    }

    void EmplaceNode(double value, double grad, std::vector<Node*> children) {
        nodes_.emplace_back(value, grad, children);
    }

    void TopologicalSort() {
        // TODO

    }

    std::vector<Node> nodes() { return nodes_; }

    void clear() { nodes_.clear(); }

    void resize(size_t size) { nodes_.resize(size); }

};

}

#endif // !GRAPH_H
