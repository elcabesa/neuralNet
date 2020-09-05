#include <algorithm>

#include "relu.h"

reluActivation::reluActivation() {
    
}

reluActivation::~reluActivation() {
    
}

double reluActivation::propagate(double input) const {
    return std::max(input, 0.0);
}

double reluActivation::derivate(double input) const {
    return input >=0 ? 1 : 1e-3;
}

const std::string reluActivation::getType() const {
    return "Relu";
}
