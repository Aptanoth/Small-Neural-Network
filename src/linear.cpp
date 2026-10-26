#include "snn.hpp"
#include "layers.hpp"

Vec Linear::forward(const Vec& in){
    this->input = in;

    return add(matVec(weights, input), bias);
}

Vec Linear::backward(const Vec& grad_output, double learning_rate){
    size_t h = this->weights.size();
    size_t w = this->weights[0].size();
    size_t i = 0;
    size_t j = 0;

    Vec input_grad = matVec(transpose(weights), grad_output);

    while(i < w){
        j = 0;
        while (j < h){
            this->weights[j][i] -= learning_rate * grad_output[j] * this->input[i];
            if (i == 0){
                this->bias[j] -= learning_rate * grad_output[j];
            }
            j++;
        }
        i++;
    }

    return input_grad;
}
