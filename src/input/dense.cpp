#include <iostream>
#include "dense.h"

DenseInput::DenseInput(const unsigned int size):Input(size) {
    _in.resize(_size, 0.0);
}

DenseInput::DenseInput(const std::vector<double> v):Input(v.size()), _in(v) {
}

DenseInput::~DenseInput() {}

void DenseInput::print() const {
    for(auto& el: _in) {
        std::cout<< el<< " ";
    }
    std::cout<<std::endl;
}

/*double& DenseInput::get(unsigned int index) {
    return _in[index];
}*/
void DenseInput::set(unsigned int index, double v) {
    _in[index] = v;
}

const double& DenseInput::get(unsigned int index) const {
    return _in[index];
}

unsigned int DenseInput::getElementNumber() const {
    return _size;
}

const std::pair<unsigned int, double> DenseInput::getElementFromIndex(unsigned int index) const {
    tempReply = std::make_pair(index, _in[index]);
    return tempReply;
}
