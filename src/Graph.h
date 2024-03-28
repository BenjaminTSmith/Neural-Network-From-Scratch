#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <set>
#include <unordered_set>
#include "Node.h"

namespace DAG {

struct Graph {
    Graph() {}

    Graph(size_t size) : nodes_(size) {}

    ~Graph() { for (auto& node_ : nodes_) delete node_; }

    Graph& operator=(const Graph& other) = default;

    Node* AddNode(Node* node) {
        if (node == nullptr)
            return nullptr;
        nodes_.push_back(node);
        return nodes_.back();
    }

    void TopologicalSort() {
        std::vector<Node*> nodes;
        std::unordered_set<Node*> visited;
        Topo(nodes_[nodes_.size() - 1], nodes, visited);
        nodes_ = nodes;
        std::reverse(nodes_.begin(), nodes_.end());
    }

    void Topo(Node* node,
              std::vector<Node*>& nodes,
              std::unordered_set<Node*>& visited) {
        if (!visited.contains(node)) {
            visited.emplace(node);
            for (auto& child_ : node->children_)  
                Topo(child_, nodes, visited); 
            nodes.push_back(node);            
        }
    }

    void TrimGraph() {
        // currently just removes duplicates with a set
        // might be unnecessary due to TopologicalSort()
        std::set<Node*> temp(nodes_.begin(), nodes_.end());
        nodes_.assign(temp.begin(), temp.end());
    }

    void BackProp() {
        nodes_[0]->grad_ = 1;
        for (const auto& node_ : nodes_)
            node_->ComputeGradients();
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
                std::cout << nodes_[i]->value_ << " : " << nodes_[i]->grad_
                    << std::endl << "  |" << std::endl;
            std::cout << nodes_[size() - 1]->value_ << " : "
                << nodes_[size() - 1]->grad_ << std::endl << std::endl;
        }
    }

private:
    std::vector<Node*> nodes_;
};

}

#endif // !GRAPH_H
