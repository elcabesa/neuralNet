#include <cassert>
#include <iostream>
#include <random>

#include "activation.h"
#include "parallelDenseLayer.h"
#include "denseLayer.h"
#include "input.h"
#include "parallelSparse.h"

ParallelDenseLayer::ParallelDenseLayer(const unsigned int number, const unsigned int inputSize, const unsigned int outputSize, std::shared_ptr<Activation> act, const unsigned int accumulatorBits, const double outScaling, const double stdDev):
    Layer{number * inputSize, number * outputSize, accumulatorBits, outScaling,  stdDev},
    _number(number),
    _layerInputSize(inputSize),
    _layerOutputSize(outputSize),
    _layerWeightNumber(_layerInputSize * _layerOutputSize),
    _act(std::move(act))
{
    _bias.resize(_layerOutputSize, 0.0);
    _weight.resize(_layerOutputSize * _layerInputSize, 1.0);

    _netOutput.resize(_outputSize, 0.0);
    
    _biasGradient.resize(_outputSize, 0.0);
    _weightGradient.resize(_layerOutputSize * _layerInputSize * _number, 0.0);
    
    _biasSumGradient.resize(_layerOutputSize, 0.0);
    _weightSumGradient.resize(_layerOutputSize * _layerInputSize, 0.0);
    
    
    _biasMovAvg.resize(_layerOutputSize, 0.0);
    _weightMovAvg.resize(_layerOutputSize * _layerInputSize, 0.0);
}

ParallelDenseLayer::~ParallelDenseLayer() {}

void ParallelDenseLayer::randomizeParams() {    
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

void ParallelDenseLayer::serialize(std::ofstream& ss) const{
    union _bb{
        int16_t d;
        char c[2];
    }bb;

    union _ww{
        int8_t d;
        char c[1];
    }ww;

    ss << "{";
    for (auto & b: _bias) {
        bb.d = std::trunc(b);
        ss.write(bb.c, 2);
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

bool ParallelDenseLayer::deserialize(std::ifstream& ss) {
    union _bb{
        int16_t d;
        char c[2];
    }bb;

    union _ww{
        int8_t d;
        char c[1];
    }ww;

    if(ss.get() != '{') {std::cout<<"DenseLayer missing {"<<std::endl;return false;}
    for (auto & b: _bias) {
        ss.read(bb.c, 2);
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

void ParallelDenseLayer::printParams() const {
    std::cout<<"weights"<<std::endl;
    for(auto& x: _weight) {std::cout<<x <<" ";} std::cout<<std::endl;
    std::cout<<"bias"<<std::endl;
    for(auto& x: _bias) {std::cout<<x <<" ";} std::cout<<std::endl;
}

void ParallelDenseLayer::_calcNetOut(const Input& input, unsigned int layer) {
    assert(input.size() == _layerInputSize);
    // copy biases to output
    for (unsigned int o = 0; o < _layerOutputSize; ++o) {
        _netOutput[o + layer * _layerOutputSize] = _getQuantizedBias(o);
    }
    

    // for each input
    unsigned int num = input.getElementNumber();
    for (unsigned int idx = 0; idx < num; ++idx) {
        auto& el = input.getElementFromIndex(idx);
        _activeFeature.insert(el.first);
        // calc all the output
        for(unsigned int o = 0; o < _layerOutputSize; ++o) {
            _netOutput[o + layer * _layerOutputSize] += el.second * _getQuantizedWeight(_calcWeightIndex(el.first, o));
        }
    } 
}

void ParallelDenseLayer::_calcOut(unsigned int layer) {
    for(unsigned int o = 0; o < _layerOutputSize; ++o) {
        unsigned int idx = o + layer * _layerOutputSize;
        // overflow warning
        if(std::abs(_netOutput[idx]) > std::pow(2, _accumulatorBits - 1)) { std::cout<<"warning acc overflow: accumulator["<<idx<<"] = " <<_netOutput[idx]<<std::endl;}
        
        _output.set(idx, _act->propagate(_netOutput[idx] / _outScaling));

        if (_quantization) {
            //output quantization
            _output.set(idx, std::trunc(_output.get(idx)));
        }
        // save min and max
        _min = std::min(_min, _output.get(idx));
        _max = std::max(_max, _output.get(idx));
    }
}

void ParallelDenseLayer::propagate(const Input& input) {
    for ( unsigned int n = 0; n < _number; ++n) {
        const ParalledSparseInput psi(input, n, _layerInputSize);
        _calcNetOut(psi, n);
        _calcOut(n);
    }

    /*std::cout<<"----------------"<<std::endl;
    for (unsigned int o = 0; o < _outputSize; ++o) {
        std::cout<<(int)(_output.get(o))<<std::endl;
    }*/
}

double ParallelDenseLayer::_getQuantizedWeight(unsigned int i) const {
    if(_quantization) {
        return std::trunc(_weight[i]);
    } else {
        return _weight[i];
    }
}

double ParallelDenseLayer::_getQuantizedBias(unsigned int i) const {
    if(_quantization) {
        return std::trunc(_bias[i]);
    } else {
        return _bias[i];
    }
}

unsigned int ParallelDenseLayer::_calcWeightIndex(const unsigned int i, const unsigned int o) const {
    assert(o + i * _layerOutputSize < _weight.size());
    return o + i * _layerOutputSize;
}

void ParallelDenseLayer::printMinMax() {
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"min "<<_min<<std::endl;
    std::cout<<"max "<<_max<<std::endl;
}

std::vector<double> ParallelDenseLayer::backPropHelper() const {
    std::vector<double> ret;
    std::cout<<"AAAAAAAAAAAAAAAAAAAAA"<<std::endl;
    return ret;    
}

double ParallelDenseLayer::getBiasSumGradient(unsigned int index) const {
    assert(index < _bias.size());
    return _biasSumGradient[index];
}

double ParallelDenseLayer::getWeightSumGradient(unsigned int index) const {
    assert(index < _weight.size());
    return _weightSumGradient[index];
}

void ParallelDenseLayer::resetSum() {
    for(auto f: _activeFeature) {
        for(unsigned int o = 0; o < _layerOutputSize; ++o) {
            unsigned int idx = _calcWeightIndex(f, o);
            _weightSumGradient[idx] = 0.0;
        }
    }
    _activeFeature.clear();
    _biasSumGradient.clear();
    _biasSumGradient.resize(_layerOutputSize, 0.0);
}


void ParallelDenseLayer::upgradeBias(double beta, double learnRate, bool rmsprop) {
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
        if(std::abs(b) > std::pow(2, 15)) {std::cout<<"ERROR, parallel bias overflow"<<i<<" "<<b<<std::endl;}
        if(b > 32767) {b = 32767;}
        if(b < -32767) {b = -32767;}
        ++i;
    }
    
}

void ParallelDenseLayer::upgradeWeight(double beta, double learnRate, double regularization, bool rmsprop) {
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
            if(std::abs(_weight[idx]) > std::pow(2, 7)) {std::cout<<"ERROR, parallel weight overflow "<<idx<<" "<<_weight[idx]<<std::endl;}
            if(_weight[idx] > 127) {_weight[idx] = 127;}
            if(_weight[idx] < -127) {_weight[idx] = -127;}
        }
    }
}

void ParallelDenseLayer::backwardPropagate(const std::vector<double>& h, const Input& input) {
    _backwardCalcBiasGradient(h);
    _backwardCalcWeightGradient(input);
    _accumulateGradients(input);
}

void ParallelDenseLayer::_accumulateGradients(const Input& input) {
    assert(input.size() == _inputSize);

    unsigned int i= 0;
    for(auto& b: _biasSumGradient) {
        for (unsigned int layer = 0; layer < _number; ++layer) {
            b += _biasGradient[i + layer * _layerOutputSize];
        }
        ++i;
    }

    for (unsigned int layer = 0; layer < _number; ++layer) {
        const ParalledSparseInput psi(input, layer , _layerInputSize);
        unsigned int num = psi.getElementNumber();
        for(unsigned int idx = 0; idx < num; ++idx) {
            auto & el = psi.getElementFromIndex(idx);
            for(unsigned int o = 0; o < _layerOutputSize; ++o) {
                unsigned int index = _calcWeightIndex(el.first, o);
                _weightSumGradient[index] += _weightGradient[index +  layer * _weight.size()];
            }
        }
    }
}

void ParallelDenseLayer::_backwardCalcBiasGradient(const std::vector<double>& h) {
    assert(h.size() == _outputSize);

    unsigned int i = 0;
    for(auto& b: _biasGradient) {
        double activationDerivate = _act->derivate(_netOutput[i]);
        b = h[i] * activationDerivate / _outScaling;
        ++i;
    }
}


void ParallelDenseLayer::_backwardCalcWeightGradient(const Input& input) {
    for (unsigned int layer = 0; layer < _number; ++layer) {
        const ParalledSparseInput psi(input, layer , _layerInputSize);
        assert(psi.size() == _layerInputSize);
        unsigned int num = psi.getElementNumber();
        for(unsigned int idx = 0; idx < num; ++idx) {
            auto & el = psi.getElementFromIndex(idx);
            for(unsigned int o = 0; o < _layerOutputSize; ++o) {
                double w = _biasGradient[o + layer * _layerOutputSize] * el.second;
                _weightGradient[_calcWeightIndex(el.first, o) + layer * _weight.size()] = w;
            }
        }
    }
}
