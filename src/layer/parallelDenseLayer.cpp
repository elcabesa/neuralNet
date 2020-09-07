#include <iostream>
#include <random>

#include "activation.h"
#include "parallelDenseLayer.h"
#include "denseLayer.h"
#include "input.h"
#include "parallelSparse.h"

ParallelDenseLayer::ParallelDenseLayer(const unsigned int number, const unsigned int inputSize, const unsigned int outputSize, std::shared_ptr<Activation> act, const double stdDev):
    Layer{number * inputSize, number * outputSize, stdDev}, _number(number)
    
{
    _bias.resize(number * outputSize, 0.0);
    _weight.resize(number * outputSize * inputSize, 1.0);
    
    
    for(unsigned int n = 0 ; n < number; ++n){
        _parallelLayers.emplace_back(DenseLayer(inputSize, outputSize, act, _stdDev));
    }
    
}

ParallelDenseLayer::~ParallelDenseLayer() {}

std::vector<double>& ParallelDenseLayer::bias() {return _bias;}
std::vector<double>& ParallelDenseLayer::weight() {return _weight;}

void ParallelDenseLayer::consolidateResult() {
    unsigned int n =0;
    for(auto& l: _parallelLayers) {
        auto & b = l.bias();
        for(unsigned int o = 0; o<l.getOutputSize(); ++o) {
            b[0] = _bias[_calcBiasIndex(n, o)];
        }
        auto & w = l.weight();
        for(unsigned int i = 0; i<l.getInputSize(); ++i) {
            for(unsigned int o = 0; o<l.getOutputSize(); ++o) {
                w[l._calcWeightIndex(i,o)] = _weight[_calcWeightIndex(n, i, o)];
            }
        }
        ++n;
    }
}

double ParallelDenseLayer::getBiasSumGradient(unsigned int index) const{
    unsigned int layerNum = index / (_outputSize / _number);
    return _parallelLayers[layerNum].getBiasSumGradient(index%(_outputSize / _number));
}
double ParallelDenseLayer::getWeightSumGradient(unsigned int index) const {
    unsigned int layerNum = index / ((_inputSize/ _number) * (_outputSize / _number));
    return _parallelLayers[layerNum].getWeightSumGradient(index%((_inputSize/ _number) * (_outputSize / _number)));
}

unsigned int ParallelDenseLayer::_calcWeightIndex(const unsigned int layer, const unsigned int i, const unsigned int o) const {
    return i + (o * _inputSize / _number) + (layer * (_inputSize / _number) * (_outputSize / _number));
}

unsigned int ParallelDenseLayer::_calcBiasIndex(const unsigned int layer, const unsigned int o) const {
    return o + (layer * (_outputSize / _number));
}

void ParallelDenseLayer::randomizeParams() {    
    unsigned int n = 0;
    for(auto& l: _parallelLayers) {
        l.randomizeParams();
        auto & b = l.bias();
        for(unsigned int o = 0; o<l.getOutputSize(); ++o) {
            _bias[_calcBiasIndex(n, o)] = b[o];
        }
        auto & w = l.weight();
        for(unsigned int i = 0; i<l.getInputSize(); ++i) {
            for(unsigned int o = 0; o<l.getOutputSize(); ++o) {
                _weight[_calcWeightIndex(n, i, o)] = w[l._calcWeightIndex(i,o)];
            }
        }
        ++n;
    }    
}

void ParallelDenseLayer::propagate(const Input& input) {
    unsigned int n= 0;
    for(auto& l: _parallelLayers) {
        const ParalledSparseInput psi(input, n, _inputSize / _number);
        l.propagate(psi);
        auto& out = l.output();
        unsigned int num = out.getElementNumber();
        for(unsigned int o = 0; o < num; ++o){
            auto el = out.getElementFromIndex(o);
            _output.set(_calcBiasIndex(n, el.first), el.second); 
        }
        ++n;
    }
    
}

void ParallelDenseLayer::printParams() const {
    unsigned int n= 0;
    for(auto& l: _parallelLayers) {
        l.printParams();
        ++n;
    }

}

std::vector<double> ParallelDenseLayer::backPropHelper() const {
    std::vector<double> ret;
    for(auto& l: _parallelLayers) {
        auto r = l.backPropHelper();
        ret.insert(ret.end(), r.begin(), r.end());
    }
    return ret;    
}

void ParallelDenseLayer::resetSum() {
    for(auto& l: _parallelLayers) {
        l.resetSum();
    } 
}

void ParallelDenseLayer::accumulateGradients(const Input& input) {
    unsigned int n= 0;
    for(auto& l: _parallelLayers) {
        const ParalledSparseInput psi(input, n , input.size() / _number);
        l.accumulateGradients(psi);
        ++n;
    }
}

void ParallelDenseLayer::backwardCalcBias(const std::vector<double>& h) {
    unsigned int n = 0;
    for(auto& l: _parallelLayers) {
        const std::vector<double> in(h.begin() + (_outputSize / _number) * n, h.begin() + (_outputSize / _number) * (n + 1));
        l.backwardCalcBias(in);
        ++n;
    }
}


void ParallelDenseLayer::backwardCalcWeight(const Input& input) {
    unsigned int n= 0;
    for(auto& l: _parallelLayers) {
        const ParalledSparseInput psi(input, n , input.size() / _number);
        l.backwardCalcWeight(psi);
        ++n;
    }
}


void ParallelDenseLayer::serialize(std::ofstream& ss) const{
    ss << "{";
    for(auto& l: _parallelLayers) {
        l.serialize(ss);
    }
    ss << "}"<<std::endl;
}

bool ParallelDenseLayer::deserialize(std::ifstream& ss) {
    //std::cout<<"DESERIALIZE PARALLEL DENSE LAYER"<<std::endl;
    if(ss.get() != '{') {std::cout<<"ParallelDenseLayer missing {"<<std::endl;return false;}
    for(auto& l :_parallelLayers) {
        if(!l.deserialize(ss)) {std::cout<<"ParallelDenseLayer internal layer error"<<std::endl;return false;}
    }
    if(ss.get() != '}') {std::cout<<"ParallelDenseLayer missing }"<<std::endl;return false;}
    if(ss.get() != '\n') {std::cout<<"DenseLayer missing line feed"<<std::endl;return false;}
    return true;
}
