#pragma once

#include <vector>
#include <random>
#include <functional>

using Vec = std::vector<double>;
using Mat = std::vector<std::vector<double>>;

double dot(const Vec& a, const Vec& b);
Vec hadamard(const Vec& a, const Vec& b);
Vec scale(const Vec& v, double s);
Vec add(const Vec& a, const Vec& b);
Vec substract(const Vec& a, const Vec& b);
Mat transpose(const Mat& m);
Vec matVec(const Mat& m, const Vec& v);
Vec randomVector(size_t size);
Mat randomMatrix(size_t rows, size_t cols);

double sigmoid (double x);
double sigmoidDerivative (double x);
double ReLu (double x);
double ReLuDerivative (double x);
Vec funcVec(const Vec& v, std::function<double (double)> f);
