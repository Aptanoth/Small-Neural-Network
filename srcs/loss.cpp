#include "snn.hpp"

double mse(const Vec& pred, const Vec& target){
	size_t size = pred.size();
	size_t i = 0;
	double sum = 0;

	while (i < size){
		sum += std::pow(pred[i] - target[i], 2);
		i++;
	}

	return sum/size;
}

Vec mseDerivative(const Vec& pred, const Vec& target){
	size_t size = pred.size();
	size_t i = 0;
	Vec res(size);


	while (i < size){
		res[i] = 2 * (pred[i] - target[i]) / size;
		i++;
	}

	return res;
}
