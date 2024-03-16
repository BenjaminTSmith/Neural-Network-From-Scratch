#ifndef VALUE_H
#define VALUE_H

#include "Graph.h"

namespace DAG {

class Value {
public:
    Graph* graph_ = nullptr;
    Node* node_ = nullptr;
    double value_ = 0;
    
    Value(double value) : value_(value) { node_ = new Node(value); }

    Value(double value, std::vector<Node*> child_nodes) : value_(value) {
        node_ = new Node(value, 0, child_nodes);
    }

    Value(const double value, Graph& graph) : value_(value), graph_(&graph) {
        node_ = new Node(value);
        graph_->AddNode(node_);
    }

    void SetGraph(Graph& graph) {
        graph_ = &graph; 
        graph_->AddNode(node_);
    }

    Value operator+(const Value& other) {
        Value ret(other.value_ + value_, {node_, other.node_});
        ret.graph_ = graph_;
        ret.node_ = graph_->AddNode(new Node(*other.node_ + *node_));
        return ret;
    }
    
    Value operator*(const Value& other) {
        Value ret(other.value_ * value_, {node_, other.node_});
        ret.graph_ = graph_;
        ret.node_ = graph_->AddNode(new Node(*other.node_ * *node_));
        return ret;
    }

    explicit operator double() { return value_; }

};

}

#endif // !VALUE_H
