#include <iostream>
#include "snn.hpp"
#include "layers.hpp"
#include "neural_network.hpp"

int main() {
    std::vector<Vec> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<Vec> Y = {{0}, {1}, {1}, {0}};

    NeuralNetwork net;
    net.add(std::make_unique<Linear>(2, 3));
    net.add(std::make_unique<Activation>(sigmoid, sigmoidDerivative));
    net.add(std::make_unique<Linear>(3, 1));
    net.add(std::make_unique<Activation>(sigmoid, sigmoidDerivative));

    net.fit(X, Y, 10000, 0.5);

    for (size_t i = 0; i < X.size(); ++i)
        std::cout << X[i][0] << " " << X[i][1] << " -> "
                  << net.predict(X[i])[0] << "  (target " << Y[i][0] << ")\n";
}
