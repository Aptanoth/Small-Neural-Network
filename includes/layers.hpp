#pragma once
#include "snn.hpp"

class Layer {
public:
    virtual ~Layer() = default;
    virtual Vec forward(const Vec& input) = 0;
    virtual Vec backward(const Vec& grad_output, double learning_rate) = 0;
};

class Activation : public Layer{
    const std::function<double(double)> f;
    const std::function<double(double)> df;
    Vec input;

    public:
        Activation(std::function<double(double)> f, std::function<double(double)> df)
                : f(f), df(df) {}

        Vec forward(const Vec& in) override;
        Vec backward(const Vec& grad_output, double /*learning_rate*/) override;
};

class Linear : public Layer {
    Mat weights;
    Vec bias;
    Vec input;

    public:
        Linear(size_t in, size_t out)
                : weights(randomMatrix(out, in)), bias(randomVector(out)) {}
        Linear(const Mat& weights, const Vec& bias)
                : weights(weights), bias(bias) {}

        const Mat& getWeights() const { return weights; }
        const Vec& getBias() const { return bias; }

        Vec forward(const Vec& in) override;
        Vec backward(const Vec& grad_output, double learning_rate) override;
};
