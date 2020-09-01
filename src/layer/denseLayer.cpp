#include <iostream>
#include <random>

#include "activation.h"
#include "denseLayer.h"
#include "input.h"

DenseLayer::DenseLayer(const unsigned int inputSize, const unsigned int outputSize, std::unique_ptr<Activation> act):
    Layer{inputSize, outputSize},
    _act(std::move(act))
    
{
    _bias.resize(outputSize, 0.0);
    _weight.resize(outputSize * inputSize, 1.0);
    
    _biasGradient.resize(outputSize, 0.0);
    _weightGradient.resize(outputSize * inputSize, 0.0);
    
    _biasSumGradient.resize(outputSize, 0.0);
    _weightSumGradient.resize(outputSize * inputSize, 0.0);
    
    _netOutput.resize(outputSize, 0.0);
}

DenseLayer::~DenseLayer() {}

void DenseLayer::calcNetOut(const Input& input) {
    _netOutput = _bias;
    for(unsigned int o = 0; o < _outputSize; ++o) {
        for(unsigned int idx = 0; idx < input.getElementNumber(); ++idx) {
            _netOutput[o] += input.getElementFromIndex(idx) * _weight[_calcWeightIndex(input.getPositionFromIndex(idx),o)];
        }
    }
}

void DenseLayer::calcOut() {
    for(unsigned int o=0; o < _outputSize; ++o) {
        _output.get(o) = _act->propagate(_netOutput[o]);
    }
}

void DenseLayer::propagate(const Input& input) {
    calcNetOut(input);
    calcOut();
}

unsigned int DenseLayer::_calcWeightIndex(const unsigned int i, const unsigned int o) const {
    // TODO invert order
    // TODO return o + i * _outputSize;
    return i + o * _inputSize;
}

std::vector<double>& DenseLayer::bias() {return _bias;}
std::vector<double>& DenseLayer::weight() {return _weight;}
std::vector<double>& DenseLayer::biasSumGradient() {return _biasSumGradient;}
std::vector<double>& DenseLayer::weightSumGradient() {return _weightSumGradient;}

void DenseLayer::randomizeParams() {
    std::random_device rnd;
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for(auto& x: _bias) {x = dist(rnd);}
    for(auto& x: _weight) {x = dist(rnd);}
}

void DenseLayer::printParams() const {
    std::cout<<"weights"<<std::endl;
    for(auto& x: _weight) {std::cout<<x <<" ";} std::cout<<std::endl;
    std::cout<<"bias"<<std::endl;
    for(auto& x: _bias) {std::cout<<x <<" ";} std::cout<<std::endl;
}

std::vector<double> DenseLayer::backPropHelper() const {
    std::vector<double> ret;
    ret.resize(_inputSize, 0.0);
    unsigned int i = 0;
    for(auto& r :ret) {
        for(unsigned int o = 0; o < _outputSize; ++o) {
            r += _biasGradient[o] * _weight[_calcWeightIndex(i,o)];
        }
        ++i;
    }
    return ret;    
}

void DenseLayer::resetSum() {
    _biasSumGradient.clear();
    _weightSumGradient.clear();
    _biasSumGradient.resize(_outputSize, 0.0);
    _weightSumGradient.resize(_outputSize * _inputSize, 0.0);
}

void DenseLayer::accumulateGradients() {
    unsigned int i= 0;
    for(auto& b: _biasSumGradient) {
        b += _biasGradient[i];
        ++i;
    }
    
    i= 0;
    for(auto& w: _weightSumGradient) {
        w += _weightGradient[i];
        ++i;
    }
}

void DenseLayer::backwardCalcBias(const std::vector<double>& h) {
    unsigned int i = 0;
    for(auto& b: _biasGradient) {
        double activationDerivate = _act->derivate(_netOutput[i]);
        b = h[i] * activationDerivate;
        ++i;
    }
}

void DenseLayer::backwardCalcWeight(const Input& prevOut) {
    for(unsigned int idx = 0; idx < prevOut.getElementNumber(); ++idx) {
        for(unsigned int o = 0; o < _outputSize; ++o) {
            double w = _biasGradient[o] * prevOut.getElementFromIndex(idx);
            _weightGradient[_calcWeightIndex(prevOut.getPositionFromIndex(idx),o)] = w;
        }
    }
}
