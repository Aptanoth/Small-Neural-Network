#include "snn.hpp"

double dot(const Vec& a, const Vec& b){
    size_t size = a.size();
    size_t i = 0;
    double res = 0;

    while (i < size){
        res += a[i] * b[i];
        i++;
    }

    return res;
}

Vec hadamard(const Vec& a, const Vec& b){
    size_t size = a.size();
    size_t i = 0;
    Vec res(size);

    while (i < size){
        res[i] = a[i] * b[i];
        i++;
    }

    return res;
}

Vec scale(const Vec& v, double s){
    size_t size = v.size();
    size_t i = 0;
    Vec res(size);

    while (i < size){
        res[i] = v[i] * s;
        i++;
    }

    return res;
}

Vec add(const Vec& a, const Vec& b){
    size_t size = a.size();
    size_t i = 0;
    Vec res(size);

    while (i < size){
        res[i] = a[i] + b[i];
        i++;
    }

    return res;
}

Vec substract(const Vec& a, const Vec& b){
    size_t size = a.size();
    size_t i = 0;
    Vec res(size);

    while (i < size){
        res[i] = a[i] - b[i];
        i++;
    }

    return res;
}

Mat transpose(const Mat& m){
    size_t h = m.size();
    size_t w = m[0].size();
    size_t i = 0;
    size_t j = 0;
    Mat res(w, Vec(h));

    while (i < h){
        j = 0;
        while (j < w){
            res[j][i] = m[i][j];
            j++;
        }
        i++;
    }

    return res;
}

Vec matVec(const Mat& m, const Vec& v){
    size_t h = m.size();
    size_t i = 0;
    Vec res(h);

    while (i < h){
        res[i] = dot(m[i], v);
        i++;
    }

    return res;
}

Vec randomVector(size_t size){
    size_t i = 0;
    Vec v(size);

    static std::mt19937 gen(time(0));
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);

    while(i < size){
        v[i] = dist(gen);
        i++;
    }

    return v;
}

Mat randomMatrix(size_t rows, size_t cols){
    size_t i = 0;
    Mat m(rows);

    while(i < rows){
        m[i] = randomVector(cols);
        i++;
    }

    return m;
}

double sigmoid(double x){
    return 1/(1 + exp(-x));
}

double sigmoidDerivative(double x){
    return sigmoid(x) * (1 - sigmoid(x));
}

double ReLu(double x){
    return x < 0 ? 0 : x;
}

double ReLuDerivative(double x){
    return x < 0 ? 0 : 1;
}

Vec funcVec(const Vec& v, std::function<double (double)> f){
    size_t size = v.size();
    size_t i = 0;
    Vec res(size);

    while (i < size){
        res[i] = f(v[i]);
        i++;
    }

    return res;
}
