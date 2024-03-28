#ifndef VALUE_H
#define VALUE_H

#include "Graph.h"

namespace DAG {

struct Value {
    Graph* graph_ = nullptr;
    Node* node_ = nullptr;
    double value_ = 0;
    
    Value() {}

    Value(double value) : value_(value), node_(new Node(value)) {}

    Value(Graph& graph) : graph_(&graph), node_(new Node) {}

    Value(double value, Graph& graph) : value_(value), graph_(&graph) {
        node_ = new Node(value);
    }

    Value(double value, std::vector<Node*> child_nodes) : value_(value) {}

    void SetGraph(Graph& graph) {
        graph_ = &graph; 
    }

    Value operator+(const Value& other) const {
        graph_->AddNode(node_);
        graph_->AddNode(other.node_);
        Value ret(other.value_ + value_, {node_, other.node_});
        ret.graph_ = graph_;
        ret.node_ = graph_->AddNode(new Node(*other.node_ + *node_));
        return ret;
    }

    Value& operator+=(const Value& other) {
        graph_->AddNode(node_);
        graph_->AddNode(other.node_);
        value_ += other.value_;
        node_ = graph_->AddNode(new Node(*other.node_ + *node_));
        return *this;
    }
    
    Value operator*(const Value& other) const {
        graph_->AddNode(node_);
        graph_->AddNode(other.node_);
        Value ret(other.value_ * value_, {node_, other.node_});
        ret.graph_ = graph_;
        ret.node_ = graph_->AddNode(new Node(*other.node_ * *node_));
        return ret;
    }

    Value& operator*=(const Value& other) {
        graph_->AddNode(node_);
        graph_->AddNode(other.node_);
        value_ *= other.value_;
        node_ = graph_->AddNode(new Node(*other.node_ * *node_));
        return *this;
    }

    Value operator/(const Value& other) const {
        graph_->AddNode(node_);
        graph_->AddNode(other.node_);
        Value ret(value_ / other.value_, {node_, other.node_});
        ret.graph_ = graph_;
        ret.node_ = graph_->AddNode(new Node(*node_ / *other.node_));
        return ret;
    }

    Value& operator/=(const Value& other) {
        graph_->AddNode(node_);
        graph_->AddNode(other.node_);
        value_ /= other.value_;
        node_ = graph_->AddNode(new Node(*node_ / *other.node_));
        return *this;
    }

    Value operator-(const Value& other) const {
        graph_->AddNode(node_);
        graph_->AddNode(other.node_);
        Value ret(value_ - other.value_, {node_, other.node_});
        ret.graph_ = graph_;
        ret.node_ = graph_->AddNode(new Node(*node_ - *other.node_));
        return ret;
    }

    // add operator-=()

    Value ReLU() const {
        graph_->AddNode(node_);
        Value ret(std::max(0.0, value_), { node_ });
        ret.graph_ = graph_;
        ret.node_ = graph_->AddNode(new Node(node_->ReLU()));
        return ret;
    }

    void SetRandom() {
        value_ = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
        node_->value_ = value_;
    }
        
    explicit operator double() const { return value_; }

};

}

#endif // !VALUE_H
