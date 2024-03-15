#ifndef VALUE_H
#define VALUE_H

#include "Graph.h"

namespace dag {

class Value {
public:
    Graph* graph_ = nullptr;
    Node* self_ = nullptr;
    double value_ = 0;
    
    Value(double value) : value_(value) { self_ = new Node(value); }

    Value(double value, std::vector<Node*> child_nodes) : value_(value) {
        self_ = new Node(value, 0, child_nodes);
    }

    void SetGraph(Graph& graph) {
        graph_ = &graph; 
        graph_->AddNode(self_);
    }

    Value operator+(const Value& other) {
        Value ret(other.value_ + value_, {self_, other.self_});
        std::cout << "here 1" << std::endl;
        ret.self_ = graph_->AddNode(*other.self_ + *self_);
        std::cout << "here 2" << std::endl;

        return ret;
    }

};

}

#endif // !VALUE_H
