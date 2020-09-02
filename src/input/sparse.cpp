#include <iostream>
#include "sparse.h"

SparseInput::SparseInput(const unsigned int size):Input(size) {
}

SparseInput::SparseInput(const std::map<unsigned int, double> v):Input(v.size()), _in(v) {
}

SparseInput::SparseInput(const std::vector<double> v):Input(v.size()) {
    unsigned int index = 0;
    for(auto& val: v) {
        _in.insert(std::pair<unsigned int, double>(index++,val));
    }
}

SparseInput::~SparseInput() {}

void SparseInput::print() const {
    for(auto& el: _in) {
        std::cout<< "{"<<el.first<< ","<<el.second<<"} ";
    }
    std::cout<<std::endl;
}

/*double& SparseInput::get(unsigned int index) {
    auto it = _in.find(index);
    if (it != _in.end()) {
        return it->second;
    }
    return _zeroInput;
}*/

void SparseInput::set(unsigned int index, double v) {
    
    _in[index] = v;
}

const double& SparseInput::get(unsigned int index) const {
    auto it = _in.find(index);
    if (it != _in.end()) {
        return it->second;
    }
    return _zeroInput;
}

unsigned int SparseInput::getElementNumber() const {
    return _in.size();
}

const std::pair<unsigned int, double> SparseInput::getElementFromIndex( unsigned int index) const {
    auto it = _in.begin();
    std::advance(it, index);
    return (*it);
}
