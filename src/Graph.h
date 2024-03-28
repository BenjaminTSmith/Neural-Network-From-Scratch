#ifndef GRAPH_H
#define GRAPH_H

#include <set>
#include <unordered_set>
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
        std::set<Node*> temp(nodes_.begin(), nodes_.end());
        nodes_.assign(temp.begin(), temp.end());
    }

    void BackProp() {
        // TODO
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
