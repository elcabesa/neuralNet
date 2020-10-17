#include <algorithm>

#include "relu.h"

reluActivation::reluActivation() {
    
}

reluActivation::~reluActivation() {
    
}

double reluActivation::propagate(double input) const {
    return std::min(std::max(input, alpha * input), saturation + alpha * (input - saturation));
}

double reluActivation::derivate(double input) const {
    return (input >= 0.0) ? ((input >= saturation) ? alpha : 1) : alpha;
}

Activation::type reluActivation::getType() const {
    return Activation::type::relu;
}
