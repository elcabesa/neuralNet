#include <algorithm>

#include "relu.h"

reluActivation::reluActivation() {
    
}

reluActivation::~reluActivation() {
    
}

double reluActivation::propagate(double input) const {
    return std::min(std::max(input, alpha * input), 1.0 + alpha * (input- 1.0));
}

double reluActivation::derivate(double input) const {
    return (input >=0) ? ((input>=1) ? alpha: 1) : alpha;
}

const std::string reluActivation::getType() const {
    return "Relu";
}
