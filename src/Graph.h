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

    Graph& operator=(const Graph& other) = default;

    std::shared_ptr<Node> AddNode(std::shared_ptr<Node> node) {
        nodes_.push_back(node);
        return nodes_.back();
    }

    void TopologicalSort() {
        std::vector<std::shared_ptr<Node>> nodes;
        std::unordered_set<std::shared_ptr<Node>> visited;
        Topo(nodes_.back(), nodes, visited);
        nodes_ = nodes;
        std::reverse(nodes_.begin(), nodes_.end());
    }

    void Topo(std::shared_ptr<Node> node,
              std::vector<std::shared_ptr<Node>>& nodes,
              std::unordered_set<std::shared_ptr<Node>>& visited) {
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

    std::vector<std::shared_ptr<Node>>& nodes() { return nodes_; }

    void clear() { nodes_.clear(); }

    void resize(size_t size) { nodes_.resize(size); }

    size_t size() { return nodes_.size(); }

    std::shared_ptr<Node> back() { return nodes_.back(); }

    std::shared_ptr<Node>& operator[](size_t idx) { return nodes_[idx]; }

    void PrintGraph() {
        if (size() != 0) {
            for (int i = 0; i < size() - 1; ++i)
                std::cout << nodes_[i]->value_ << " : " << nodes_[i]->grad_
                    << std::endl;
            std::cout << nodes_[size() - 1]->value_ << " : "
                << nodes_[size() - 1]->grad_ << std::endl;
        }
    }

private:
    std::vector<std::shared_ptr<Node>> nodes_;
};

}

#endif // !GRAPH_H
