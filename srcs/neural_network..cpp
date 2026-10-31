#include "snn.hpp"
#include "layers.hpp"
#include <iostream>
#include "neural_network.hpp"

void NeuralNetwork::add(std::unique_ptr<Layer> layer){
    this->layers.push_back(std::move(layer));
}

Vec NeuralNetwork::predict(const Vec& input){
    size_t size = this->layers.size();
    size_t i = 1;

    Vec current = this->layers[0]->forward(input);
    while (i < size){
        current = this->layers[i]->forward(current);
        i++;
    }
    return current;
}

void NeuralNetwork::fit(const std::vector<Vec>& X, const std::vector<Vec>& Y, size_t training_len, double lr){
    size_t i = 0;
    size_t j = 0;
    size_t data_size = X.size();
    double loss = 0;

    while (i < training_len){
        loss = 0;
        j = 0;

        while (j < data_size){
            //forward
            Vec pred = predict(X[j]);

            loss += mse(pred, Y[j]);
            //first gradient calculation to feed through the loop
            Vec grad = mseDerivative(pred, Y[j]);
            //bacward through every layer
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                grad = (*it)->backward(grad, lr);
            j++;
        }

        if (i % 100 == 0)
            std::cout << "epoch " << i << "  loss " << loss / data_size << "\n";
        i++;
    }
}
