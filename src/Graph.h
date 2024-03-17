#ifndef GRAPH_H
#define GRAPH_H

#include "Node.h"

namespace DAG {

struct Graph {
    Graph() {}

    Graph(size_t size) : nodes_(size) {}

    Graph& operator=(const Graph& other) = default;

    Node* AddNode(Node* node) {
        nodes_.push_back(node);
        return nodes_.back();
    }

    void EmplaceNode(double value, double grad, std::vector<Node*> children) {
        // nodes_.emplace_back(value, grad, children);
    }

    void TopologicalSort() {
        // TODO
    }

    void BackProp() {
        // TODO
        TopologicalSort();
        nodes_[nodes_.size() - 1]->grad_ = 1;
        for (int i = nodes_.size() - 1; i >= 0; --i) {
            nodes_[i]->ComputeGradients();
        }
    }

    std::vector<Node*>& nodes() { return nodes_; }

    void clear() { nodes_.clear(); }

    void resize(size_t size) { nodes_.resize(size); }

    size_t size() { return nodes_.size(); }

    Node* back() { return nodes_.back(); }

    Node*& operator[](size_t idx) { return nodes_[idx]; }

    void PrintGraph() {
        if (size() != 0) {
            for (int i = 0; i < size() - 1; ++i)
                std::cout << nodes_[i]->value_ << std::endl << '|' << std::endl;
            std::cout << nodes_[size() - 1]->value_ << std::endl << std::endl;
        }
    }

private:
    std::vector<Node*> nodes_;
};

}

#endif // !GRAPH_H
