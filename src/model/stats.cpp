#include <iostream>

#include "model.h"
#include "stats.h"

Stats::Stats(Model& m): _m(m), _examples(0) {

    for(unsigned int l = 0; l< _m.getLayerCount(); ++l) {
        unsigned int size = _m.getLayer(l).getOutputSize();
        _dyingReluCounter.push_back(std::vector<unsigned long long int>(size));
    }
}

void Stats::update() {
    ++_examples;
    for (unsigned int l = 0; l< _m.getLayerCount(); ++l) {
        const auto& layer = _m.getLayer(l);
        if (layer.getActivatinType() == Activation::type::relu) {
            for (unsigned int i = 0; i < layer.getOutputSize(); ++i) {
                if (layer.getOutput(i) > 0 && layer.getOutput(i) < 127) {
                    ++_dyingReluCounter[l][i];
                }
            }
        } else {
            for (unsigned int i = 0; i < layer.getOutputSize(); ++i) {
                _dyingReluCounter[l][i] = _examples;
            }
        }
    }
}

void Stats::print() const {
    std::cout<<"dying RELU stats"<<std::endl;
    for (unsigned int l = 0; l< _m.getLayerCount(); ++l) {
        std::cout<<"layer "<<l<<std::endl;
        const auto& layer = _m.getLayer(l);
        unsigned int death = 0;
        unsigned int dying = 0;
        for (unsigned int i = 0; i < layer.getOutputSize(); ++i) {
            if (_dyingReluCounter[l][i] == 0) {++death;}
            if (_dyingReluCounter[l][i] <= _examples * 0.05) {++dying;}
            //std::cout<< _dyingReluCounter[l][i]<<std::endl;
        }
        std::cout<< death <<" death neurons / "<<layer.getOutputSize()<< " (" << death * 100.0 / layer.getOutputSize()<<"%)"<<std::endl;
        std::cout<< dying <<" dying neurons / "<<layer.getOutputSize()<< " (" << dying * 100.0 / layer.getOutputSize() <<"%)"<<std::endl;
    }
}