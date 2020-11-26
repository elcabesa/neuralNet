#include <cassert>
#include <iostream>
#include <random>

#include "activation.h"
#include "denseLayer.h"
#include "input.h"

DenseLayer::DenseLayer(const unsigned int inputSize, const unsigned int outputSize, std::shared_ptr<Activation> act, const double outScaling, const double stdDev):
    Layer{inputSize, outputSize, stdDev, act}
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
        
        _output.set(o, _act->propagate(_netOutput[o]));

        // save min and max
        _min = std::min(_min, _output.get(o));
        _max = std::max(_max, _output.get(o));
    }
}

void DenseLayer::propagate(const Input& input) {
    _calcNetOut(input);
    _calcOut();
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

    if(_outputSize != 1) {
        std::random_device rnd;
        std::normal_distribution<double> dist(0.0, sqrt(1.0 / _inputSize));
        for (auto& x: _weight) {x = dist(rnd);}

        for(unsigned int o = 0; o < _outputSize; ++o) {
            double sum = 0.0;
            for(unsigned int i = 0; i < _inputSize; ++i) {
                sum += _weight[_calcWeightIndex(i, o)];
            }
            _bias[o] = 0.5 - 0.5 * sum;
        }
    }
    else {
        for (auto& x: _weight) {x = 0.0;}
        for (auto& x: _bias) {x = 0.0;}
    }
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
        double activationDerivate = _act->derivate(_netOutput[i]);
        b = h[i] * activationDerivate;
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
            b -= gradBias * (learnRate / sqrt(_biasMovAvg[i] + 1e-8));
        } else {
            /*_biasMovAvg[i] += gradBias * gradBias;*/
            b -= gradBias * learnRate /*/ std::sqrt(_biasMovAvg[i] + 1e-8)*/;
        }
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
                _weight[idx] = (regularization * _weight[idx]) - gradWeight * (learnRate / sqrt(_weightMovAvg[idx] + 1e-8));
            }
            else {
                /*_weightMovAvg[idx] += gradWeight * gradWeight;*/
                _weight[idx] = (regularization * _weight[idx]) - gradWeight * learnRate/* / std::sqrt(_weightMovAvg[idx] + 1e-8)*/;
            }
        }
    }
}

void DenseLayer::serialize(std::ofstream& ss) const{
    double min = 1e8;
    double max = -1e8;
    union _bb{
        int32_t d;
        char c[4];
    }bb;

    union _ww{
        int8_t d;
        char c[1];
    }ww;
    std::cout<<"SERIALIZE LAYER"<<std::endl;

    ss << "{";
    double bgain = _outputSize == 1 ? 30000.0 / 4.0 : 127.0 * 64.0;
    for (auto & b: _bias) {
        double _b = b * bgain;
        max = std::max(_b, max);
        min = std::min(_b, min);
        bb.d = std::round(_b);
        ss.write(bb.c, 4);
        //ss<<", ";
    }
    std::cout<<" layer bias min: "<<min<< " max: "<< max<<std::endl;


    min = 1e8;
    max = -1e8;
    ss <<std::endl;
    double wgain = _outputSize == 1 ? 30000.0 / 4.0 / 127.0 : 64.0;
    for (auto & w: _weight) {
        double _w = w * wgain;
        max = std::max(_w, max);
        min = std::min(_w, min);
        ww.d = std::round(_w);
        ss.write(ww.c, 1);
        //ss<<", ";
    }
    std::cout<<" layer weight min: "<<min<< " max: "<< max<<std::endl;

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
    double bgain = _outputSize == 1 ? 30000.0 / 4.0 : 127.0 * 64.0;
    if(ss.get() != '{') {std::cout<<"DenseLayer missing {"<<std::endl;return false;}
    for (auto & b: _bias) {
        ss.read(bb.c, 4);
        b = bb.d / bgain;
        //if(ss.get() != ',') {std::cout<<"DenseLayer missing ,"<<std::endl;return false;}
        //if(ss.get() != ' ') {std::cout<<"DenseLayer missing space"<<std::endl;return false;}
    }
    if(ss.get() != '\n') {std::cout<<"DenseLayer missing line feed"<<std::endl;return false;}
    double wgain = _outputSize == 1 ? 30000.0 / 4.0 / 127.0 : 64.0;
    for (auto & w: _weight) {
        ss.read(ww.c, 1);
        w = ww.d / wgain;
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
        double wgain = _outputSize == 1 ? 30000.0 / 4.0 / 127.0 : 64.0;
        return std::round(wgain * _weight[i]) / wgain;
    } else {
        return _weight[i];
    }
}

double DenseLayer::_getQuantizedBias(unsigned int i) const {
    if(_quantization) {
        double bgain = _outputSize == 1 ? 30000.0 / 4.0 : 127.0 * 64.0;
        return std::round(bgain * _bias[i]) / bgain;
    } else {
        return _bias[i];
    }
}