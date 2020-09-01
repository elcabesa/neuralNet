#include <iostream>
#include "input.h"

Input::Input(const unsigned int size): _size(size) {
    _in.resize(_size, 0.0);
}

Input::Input(const std::vector<double> v):_size(v.size()), _in(v) {
}

Input::~Input() {}

void Input::print() const {
    for(auto& el: _in) {
        std::cout<< el<< " ";
    }
    std::cout<<std::endl;
}

double& Input::get(unsigned int index) {
    return _in[index];
}

const double& Input::get(unsigned int index) const {
    return _in[index];
}

unsigned int Input::getElementNumber() const {
    return _size;
}

double& Input::getElementFromIndex( unsigned int index) {
    return _in[index];
}
const double& Input::getElementFromIndex( unsigned int index) const {
    return _in[index];
}

unsigned int Input::getPositionFromIndex(unsigned int index) const {
    return index;
}


