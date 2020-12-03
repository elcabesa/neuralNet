#include <cmath>
#include <iostream>

#include "variance.h"

VarianceCalculator::VarianceCalculator() {
    reset();
}
void VarianceCalculator::reset() {
    _n = 0;
    _ex = 0.0;
    _ex2 = 0.0;
}

void VarianceCalculator::addValue(double x) {
    ++_n;
    _ex += x;
    _ex2 += x * x;
}

double VarianceCalculator::getMean() const {
    return _ex / _n;
}

double VarianceCalculator::getVariance() const {
    if( _n < 2) return 0;
    return (_ex2 - (_ex * _ex) / _n) / (_n -1);
}

double VarianceCalculator::getstdDev() const {
    return sqrt(getVariance());
}

void VarianceCalculator::print() {
    std::cout <<getMean() <<" "<<getVariance()<<" "<<getstdDev()<< std::endl;
    reset();
}

