#include "snn.hpp"
#include "layers.hpp"

Vec Activation::forward(const Vec& in){
    this->input = in;
    return funcVec(in, this->f);
}

Vec Activation::backward(const Vec& grad_output, double){
    return hadamard(funcVec(this->input, this->df), grad_output);
}
