#ifndef GRAPH_H
#define GRAPH_H

#include "Node.h"

namespace dag {

class Graph {
public:
    std::vector<Node> nodes_;
    
    Graph() {}

    Graph(size_t size) : nodes_(size) {}

    Graph& operator=(const Graph& other) {
        nodes_ = other.nodes_;
        return *this;
    }

    // returns pointer to added node
    Node* AddNode(const Node& node) {
        std::cout << "this one" << std::endl;
        std::cout << nodes_.size() << std::endl;
        nodes_.push_back(std::move(node));
        std::cout << "no this one" << std::endl;
        return &nodes_.back();
    }

    // heap allocated node. only use with Nodes created by new
    Node* AddNode(Node* node) {
        nodes_.push_back(*node);
        return &nodes_.back();
    }

    void EmplaceNode(double value, double grad, std::vector<Node*> children) {
        nodes_.emplace_back(value, grad, children);
    }

    void TopologicalSort() {
        // TODO

    }

    std::vector<Node>& nodes() { return nodes_; }

    void clear() { nodes_.clear(); }

    void resize(size_t size) { nodes_.resize(size); }

    void PrintGraph() {
        for (const auto& node : nodes_)
            std::cout << node.value_ << std::endl << '|' << std::endl;
    }

};

}

#endif // !GRAPH_H
