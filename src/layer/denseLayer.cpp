#include <cassert>
#include <iostream>
#include <random>

#include "activation.h"
#include "denseLayer.h"
#include "input.h"

DenseLayer::DenseLayer(const unsigned int inputSize, const unsigned int outputSize, std::shared_ptr<Activation> act, const double stdDev):
    Layer{inputSize, outputSize, stdDev},
    _act(std::move(act))
    
{
    _bias.resize(outputSize, 0.0);
    _weight.resize(outputSize * inputSize, 1.0);
    _quantizedWeight.resize(outputSize * inputSize, 1.0);
    
    _biasGradient.resize(outputSize, 0.0);
    _weightGradient.resize(outputSize * inputSize, 0.0);
    
    _biasSumGradient.resize(outputSize, 0.0);
    _weightSumGradient.resize(outputSize * inputSize, 0.0);
    
    _netOutput.resize(outputSize, 0.0);
    
    _biasMovAvg.resize(outputSize, 0.0);
    _weightMovAvg.resize(outputSize * inputSize, 0.0);
}

DenseLayer::~DenseLayer() {}

void DenseLayer::calcNetOut(const Input& input) {
    assert(input.size() == _inputSize);
    _netOutput = _bias;
    unsigned int num = input.getElementNumber();
    for(unsigned int idx = 0; idx < num; ++idx) {
        auto& el = input.getElementFromIndex(idx);
        _activeFeature.insert(el.first);
        for(unsigned int o = 0; o < _outputSize; ++o) {
            _netOutput[o] += el.second * _weight[_calcWeightIndex(el.first,o)];
        }
    }
    
}

void DenseLayer::calcOut() {
    //unsigned int zeros = 0;
    //unsigned int negative = 0;
    for(unsigned int o=0; o < _outputSize; ++o) {
        _output.set(o, _act->propagate(_netOutput[o]));
        if(_quantization) {
            //output quantization
            _output.set(o,(std::lround(_output.get(o) * 255)/255));
        }
        _output.get(o);
        _min = std::min(_min,  _output.get(o));
        _max = std::max(_max,  _output.get(o));
        /*if(_netOutput[o]<0 ){
            ++negative;
        }*/
        /*if (_output.get(o) != _netOutput[o]) {
            ++zeros;
        }*/
    }
    //if(zeros*100 /_outputSize >75) {std::cout<<"WARNING high death neurons ("<<zeros<<"/"<<_outputSize<<")"<<std::endl;}
    //std::cout<<"NEGATIVE/TOTAL:"<<negative<<" "<<_outputSize<<std::endl;
}

void DenseLayer::propagate(const Input& input) {
    calcNetOut(input);
    calcOut();
}

unsigned int DenseLayer::_calcWeightIndex(const unsigned int i, const unsigned int o) const {
    assert(o + i * _outputSize < _weight.size());
    return o + i * _outputSize;
    //return i + o * _inputSize;
}

std::vector<double>& DenseLayer::bias() {return _bias;}
std::vector<double>& DenseLayer::weight() {return _weight;}

void DenseLayer::consolidateResult() {}

double DenseLayer::getBiasSumGradient(unsigned int index) const{
    assert(index < _bias.size());
    return _biasSumGradient[index];
}
double DenseLayer::getWeightSumGradient(unsigned int index) const{
    assert(index < _weight.size());
    return _weightSumGradient[index];
}

void DenseLayer::randomizeParams() {
    double stdDev = _stdDev;
    if( stdDev == 0.0) {
        stdDev = sqrt(2.0 / _inputSize);
    }
    //std::cout<<"std dev "<<stdDev<<std::endl;
    std::random_device rnd;
    /*std::normal_distribution<double> dist(0.0, 1.0);*/
    std::normal_distribution<double> dist2(0.0, stdDev);
    
    /*for(auto& x: _bias) {x = dist(rnd);}*/ 
    for(auto& x: _weight) {x = dist2(rnd);}
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
    for(auto f: _activeFeature) {
        for(unsigned int o = 0; o < _outputSize; ++o) {
            unsigned int idx = _calcWeightIndex(f, o);
            _weightSumGradient[idx] = 0.0;
        }
    }
    //for(auto& w:_weightSumGradient) { if(w!= 0.0) {std::cout<<"ERRRRRORE"<<std::endl;exit(0);}}
    _activeFeature.clear();
    _biasSumGradient.clear();
    _biasSumGradient.resize(_outputSize, 0.0);
}

void DenseLayer::accumulateGradients(const Input& input) {
    unsigned int i= 0;
    for(auto& b: _biasSumGradient) {
        b += _biasGradient[i];
        ++i;
    }
    
    unsigned int num = input.getElementNumber();
    for(unsigned int idx = 0; idx < num; ++idx) {
        auto & el = input.getElementFromIndex(idx);
        for(unsigned int o = 0; o < _outputSize; ++o) {
            unsigned int index = _calcWeightIndex(el.first,o);
            _weightSumGradient[index] += _weightGradient[index];
        }
    }
}

void DenseLayer::backwardCalcBiasGradient(const std::vector<double>& h) {
    assert(h.size() == _outputSize);
    unsigned int i = 0;
    for(auto& b: _biasGradient) {
        double activationDerivate = _act->derivate(_netOutput[i]);
        b = h[i] * activationDerivate;
        ++i;
    }
}

void DenseLayer::backwardCalcWeightGradient(const Input& input) {
    assert(input.size() == _inputSize);
    unsigned int num = input.getElementNumber();
    for(unsigned int idx = 0; idx < num; ++idx) {
        auto & el = input.getElementFromIndex(idx);
        for(unsigned int o = 0; o < _outputSize; ++o) {
            double w = _biasGradient[o] * el.second;
            _weightGradient[_calcWeightIndex(el.first,o)] = w;
        }
    }
}

void DenseLayer::upgradeBias(double beta, double learnRate) {
    double beta2 = (1.0 - beta);
    
    for(auto& bma: _biasMovAvg){
        bma = beta * bma;
    }

    
    unsigned int i = 0;
    //std::cout<<"--------------"<<std::endl;
    for(auto& b: _bias){
        double gradBias = getBiasSumGradient(i);
        _biasMovAvg[i] += beta2 * gradBias * gradBias;
        b -= gradBias * (learnRate / sqrt(_biasMovAvg[i] /*+ 1e-8*/));
        //std::cout<<(i+1)<<": "<<b<<" "<<gradBias<<" "<<(gradBias / b * 100)<<"%"<<std::endl;
        //b -= gradBias * (learnRate);
        ++i;
    }
    
}

void DenseLayer::upgradeWeight(double beta, double learnRate, double regularization) {
    double beta2 = (1.0 - beta);
    
    for(auto& wma: _weightMovAvg){
        wma = beta * wma;
    }
    
    //std::cout<<"ACTIVE FEATURE SIZE: "<<_activeFeature.size()<<std::endl;
    for(auto f: _activeFeature) {
        for(unsigned int o = 0; o < _outputSize; ++o) {
            unsigned int idx = _calcWeightIndex(f, o);
            double gradWeight = getWeightSumGradient(idx);
            /*if(gradWeight == 0.0) {
                std::cout<<"GRAD ERROR"<<std::endl;
            }*/
            //-----------------------------
            // this should be done for every element, but I did it only for active features to speedup
            //_weightMovAvg[idx] *= beta;
            //-----------------------------
            _weightMovAvg[idx] += beta2 * gradWeight * gradWeight;
            _weight[idx] = (regularization * _weight[idx] ) - gradWeight * (learnRate / sqrt(_weightMovAvg[idx] /*+ 1e-8*/));
            //_weight[idx] -= gradWeight * (learnRate);
        }
    }
}

void DenseLayer::serialize(std::ofstream& ss) const{
    union un{
        double d;
        char c[8];
    }u;
    ss << "{";
    for( auto & b: _bias) {
        u.d = b; 
        ss.write(u.c, 8);
        ss<<", ";
    }
    ss <<std::endl;
    for( auto & w: _weight) {
        u.d = w; 
        ss.write(u.c, 8);
        ss<<", ";
    }
    
    ss << "}"<<std::endl;
}

bool DenseLayer::deserialize(std::ifstream& ss) {
    //std::cout<<"DESERIALIZE DENSE LAYER"<<std::endl;
    union un{
        double d;
        char c[8];
    }u;
    if(ss.get() != '{') {std::cout<<"DenseLayer missing {"<<std::endl;return false;}
    for( auto & b: _bias) {
        ss.read(u.c, 8);
        b = u.d;
        if(ss.get() != ',') {std::cout<<"DenseLayer missing ,"<<std::endl;return false;} 
        if(ss.get() != ' ') {std::cout<<"DenseLayer missing space"<<std::endl;return false;}
    }
    if(ss.get() != '\n') {std::cout<<"DenseLayer missing line feed"<<std::endl;return false;}
    for( auto & w: _weight) {
        ss.read(u.c, 8);
        w = u.d;
        if(ss.get() != ',') {std::cout<<"DenseLayer missing ,"<<std::endl;return false;} 
        if(ss.get() != ' ') {std::cout<<"DenseLayer missing space"<<std::endl;return false;}
    }
    if(ss.get() != '}') {std::cout<<"DenseLayer missing }"<<std::endl;return false;} 
    if(ss.get() != '\n') {std::cout<<"DenseLayer missing line feed"<<std::endl;return false;}
    return true;
}

void DenseLayer::printMinMax() {
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"min "<<_min<<std::endl;
    std::cout<<"max "<<_max<<std::endl;
}