#include <iostream>
#include <random>

#include "activation.h"
#include "denseLayer.h"
#include "input.h"

DenseLayer::DenseLayer(const unsigned int inputSize, const unsigned int outputSize, std::shared_ptr<Activation> act):
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
            auto& el = input.getElementFromIndex(idx);
            _netOutput[o] += el.second * _weight[_calcWeightIndex(el.first,o)];
        }
    }
}

void DenseLayer::calcOut() {
    for(unsigned int o=0; o < _outputSize; ++o) {
        _output.set(o, _act->propagate(_netOutput[o]));
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
            auto & el = prevOut.getElementFromIndex(idx);
            double w = _biasGradient[o] * el.second;
            _weightGradient[_calcWeightIndex(el.first,o)] = w;
        }
    }
}

void DenseLayer::serialize(std::ofstream& ss) const{
    ss << "{";
    for( auto & b: _bias) {
        ss<<b<<", ";
    }
    ss <<std::endl;
    for( auto & w: _weight) {
        ss<<w<<", ";
    }
    
    ss << "}"<<std::endl;
}

bool DenseLayer::deserialize(std::ifstream& ss) {
    //std::cout<<"DESERIALIZE DENSE LAYER"<<std::endl;
    if(ss.get() != '{') {std::cout<<"DenseLayer missing {"<<std::endl;return false;}
    for( auto & b: _bias) {
        char data[256];
        ss.get(data, 256, ',');
        b = std::stod(data);
        if(ss.get() != ',') {std::cout<<"DenseLayer missing ,"<<std::endl;return false;} 
        if(ss.get() != ' ') {std::cout<<"DenseLayer missing space"<<std::endl;return false;}
    }
    if(ss.get() != '\n') {std::cout<<"DenseLayer missing line feed"<<std::endl;return false;}
    for( auto & w: _weight) {
        char data[256];
        ss.get(data, 256, ',');
        w = std::stod(data);
        if(ss.get() != ',') {std::cout<<"DenseLayer missing ,"<<std::endl;return false;} 
        if(ss.get() != ' ') {std::cout<<"DenseLayer missing space"<<std::endl;return false;}
    }
    if(ss.get() != '}') {std::cout<<"DenseLayer missing }"<<std::endl;return false;} 
    if(ss.get() != '\n') {std::cout<<"DenseLayer missing line feed"<<std::endl;return false;}
    return true;
}
