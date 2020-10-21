#include <cassert>
#include <iostream>
#include <random>

#include "activation.h"
#include "denseLayer.h"
#include "input.h"

DenseLayer::DenseLayer(const unsigned int inputSize, const unsigned int outputSize, std::shared_ptr<Activation> act, const unsigned int accumulatorBits, const double outScaling, const double stdDev):
    Layer{inputSize, outputSize, accumulatorBits, outScaling, stdDev},
    _act(std::move(act))
{
    _bias.resize(outputSize, 0.0);
    _weight.resize(outputSize * inputSize, 1.0);
    
    _biasGradient.resize(outputSize, 0.0);
    _weightGradient.resize(outputSize * inputSize, 0.0);
    
    _biasSumGradient.resize(outputSize, 0.0);
    _weightSumGradient.resize(outputSize * inputSize, 0.0);
    
    _netOutput.resize(outputSize, 0.0);
    
    _biasMovAvg.resize(outputSize, 0.0);
    _weightMovAvg.resize(outputSize * inputSize, 0.0);
}

DenseLayer::~DenseLayer() {}

void DenseLayer::_calcNetOut(const Input& input) {
    assert(input.size() == _inputSize);
    for(unsigned int o = 0; o < _outputSize; ++o) {
        _netOutput[o] = _getQuantizedBias(o);
    }
    
    unsigned int num = input.getElementNumber();
    for(unsigned int idx = 0; idx < num; ++idx) {
        auto& el = input.getElementFromIndex(idx);
        _activeFeature.insert(el.first);
        for(unsigned int o = 0; o < _outputSize; ++o) {
            _netOutput[o] += el.second * _getQuantizedWeight(_calcWeightIndex(el.first,o));
        }
    }
    
    
}

void DenseLayer::_calcOut() {
    for(unsigned int o=0; o < _outputSize; ++o) {
        // overflow warning
        if(std::abs(_netOutput[o]) > std::pow(2, _accumulatorBits - 1)) { std::cout<<"warning acc overflow: accumulator["<<o<<"] = " <<_netOutput[o]<<std::endl;}
        
        _output.set(o, _act->propagate(_netOutput[o] / _outScaling));

        if (_quantization) {
            //output quantization
            _output.set(o, std::trunc(_output.get(o)));
        }
        // save min and max
        _min = std::min(_min, _output.get(o));
        _max = std::max(_max, _output.get(o));
    }
}

void DenseLayer::propagate(const Input& input) {
    _calcNetOut(input);
    _calcOut();

    /*std::cout<<"----------------"<<std::endl;
    for (unsigned int o = 0; o < _outputSize; ++o) {
        std::cout<<(int)(_output.get(o))<<std::endl;
    }*/
}

unsigned int DenseLayer::_calcWeightIndex(const unsigned int i, const unsigned int o) const {
    assert(o + i * _outputSize < _weight.size());
    return o + i * _outputSize;
}

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
        stdDev = 25 * sqrt(2.0 / _inputSize);
    }

    std::random_device rnd;
    std::normal_distribution<double> dist(0.0, 25);
    std::normal_distribution<double> dist2(0.0, stdDev);
    
    for (auto& x: _bias) {x = dist(rnd);}
    for (auto& x: _weight) {x = dist2(rnd);}
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
            r += _biasGradient[o] * _getQuantizedWeight(_calcWeightIndex(i,o));
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
    _activeFeature.clear();
    _biasSumGradient.clear();
    _biasSumGradient.resize(_outputSize, 0.0);
}

void DenseLayer::backwardPropagate(const std::vector<double>& h, const Input& input) {
    _backwardCalcBiasGradient(h);
    _backwardCalcWeightGradient(input);
    _accumulateGradients(input);
}

void DenseLayer::_accumulateGradients(const Input& input) {
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

void DenseLayer::_backwardCalcBiasGradient(const std::vector<double>& h) {
    assert(h.size() == _outputSize);
    unsigned int i = 0;
    for(auto& b: _biasGradient) {
        double activationDerivate = _act->derivate(_netOutput[i] / _outScaling);
        b = h[i] * activationDerivate / _outScaling;
        ++i;
    }
}

void DenseLayer::_backwardCalcWeightGradient(const Input& input) {
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

void DenseLayer::upgradeBias(double beta, double learnRate, bool rmsprop) {
    double beta2 = (1.0 - beta);
    if (rmsprop) {
        for(auto& bma: _biasMovAvg){
            bma = beta * bma;
        }
    }

    unsigned int i = 0;
    for (auto& b: _bias) {
        double gradBias = getBiasSumGradient(i);
        if (rmsprop) {
            _biasMovAvg[i] += beta2 * gradBias * gradBias;
            b -= gradBias * (learnRate / sqrt(_biasMovAvg[i] /*+ 1e-8*/));
        } else {
            b -= gradBias * learnRate;
        }
        if(std::abs(b) > std::pow(2, 31)) {std::cout<<"ERROR, bias overflow"<<i<<" "<<b<<" ["<<_inputSize<<"*"<<_outputSize<<"]"<<std::endl;}
        if(b > 2147483647) {b = 2147483647;}
        if(b < -2147483647) {b = -2147483647;}
        ++i;
    }
}

void DenseLayer::upgradeWeight(double beta, double learnRate, double regularization, bool rmsprop) {
    double beta2 = (1.0 - beta);
    
    if (rmsprop) {
        for (auto& wma: _weightMovAvg) {
            wma = beta * wma;
        }
    }

    for (auto f: _activeFeature) {
        for(unsigned int o = 0; o < _outputSize; ++o) {
            unsigned int idx = _calcWeightIndex(f, o);
            double gradWeight = getWeightSumGradient(idx);
            //-----------------------------
            // this should be done for every element, but I did it only for active features to speedup
            //_weightMovAvg[idx] *= beta;
            //-----------------------------
            if (rmsprop) {
                _weightMovAvg[idx] += beta2 * gradWeight * gradWeight;
                _weight[idx] = (regularization * _weight[idx]) - gradWeight * (learnRate / sqrt(_weightMovAvg[idx] /*+ 1e-8*/));
            }
            else {
                _weight[idx] = (regularization * _weight[idx]) - gradWeight * learnRate;
            }
            if(std::abs(_weight[idx]) > std::pow(2, 7)) {std::cout<<"ERROR, weight overflow "<<idx<<" "<<_weight[idx]<<" ["<<_inputSize<<"*"<<_outputSize<<"]"<<std::endl;}
            if(_weight[idx] > 127) {_weight[idx] = 127;}
            if(_weight[idx] < -127) {_weight[idx] = -127;}
        }
    }
}

void DenseLayer::serialize(std::ofstream& ss) const{
    union _bb{
        int32_t d;
        char c[4];
    }bb;

    union _ww{
        int8_t d;
        char c[1];
    }ww;

    ss << "{";
    for (auto & b: _bias) {
        bb.d = std::trunc(b);
        ss.write(bb.c, 4);
        //ss<<", ";
    }
    ss <<std::endl;
    for (auto & w: _weight) {
        ww.d = std::trunc(w);
        ss.write(ww.c, 1);
        //ss<<", ";
    }

    ss << "}"<<std::endl;
}

bool DenseLayer::deserialize(std::ifstream& ss) {
    union _bb{
        int32_t d;
        char c[4];
    }bb;

    union _ww{
        int8_t d;
        char c[1];
    }ww;

    if(ss.get() != '{') {std::cout<<"DenseLayer missing {"<<std::endl;return false;}
    for (auto & b: _bias) {
        ss.read(bb.c, 4);
        b = bb.d;
        //if(ss.get() != ',') {std::cout<<"DenseLayer missing ,"<<std::endl;return false;}
        //if(ss.get() != ' ') {std::cout<<"DenseLayer missing space"<<std::endl;return false;}
    }
    if(ss.get() != '\n') {std::cout<<"DenseLayer missing line feed"<<std::endl;return false;}
    for (auto & w: _weight) {
        ss.read(ww.c, 1);
        w = ww.d;
        //if(ss.get() != ',') {std::cout<<"DenseLayer missing ,"<<std::endl;return false;}
        //if(ss.get() != ' ') {std::cout<<"DenseLayer missing space"<<std::endl;return false;}
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

double DenseLayer::_getQuantizedWeight(unsigned int i) const {
    if(_quantization) {
        return std::trunc(_weight[i]);
    } else {
        return _weight[i];
    }
}

double DenseLayer::_getQuantizedBias(unsigned int i) const {
    if(_quantization) {
        return std::trunc(_bias[i]);
    } else {
        return _bias[i];
    }
}