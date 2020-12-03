#include <cmath>
#include <iostream>

#include "cost.h"

Cost::Cost() {}

Cost::~Cost() {}

double Cost::calc(const double out, const double label) const {
    //std::cout<<"LABEL "<<label<<" OUT "<<out<<std::endl;
    return std::pow((out - label), 2.0) / 2.0;
    /*double sigLabel = sigmoid(label);
    double sigOut = sigmoid(out);
    //std::cout<<label<<"-"<<out<<std::endl;
    //std::cout<<sigLabel<<" "<<sigOut<<std::endl;
    return - (sigLabel * std::log(sigOut)  + (1 - sigLabel) * std::log(1 - sigOut));*/
}

double Cost::derivate(const double out, const double label) const {
    //std::cout<<"derivate:"<<(sigmoid(out) - sigmoid(label)) * scaleFactor<<std::endl;
    //std::cout<<"out:"<<out<<" label:"<<label<<std::endl;
    //std::cout<<"sigmoid(out):"<<sigmoid(out)<<" sigmoid(label):"<<sigmoid(label)<<std::endl;
    //std::cout<<"out- label:"<<out-label<<" derivate:"<<(sigmoid(out) - sigmoid(label))* multiplier<<std::endl;
    //return (sigmoid(out) - sigmoid(label))* multiplier;
    return out - label;
}

double Cost::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x * scaleFactor));
}


