#include "snn.hpp"

class NN {
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add(std::unique_ptr<Layer> layer);
    Vec predict(const Vec& input);
    void fit(const std::vector<Vec>& X, const std::vector<Vec>& Y, size_t epochs, double lr);
};
