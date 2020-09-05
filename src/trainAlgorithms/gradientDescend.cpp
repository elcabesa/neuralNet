#include <chrono>
#include <iostream>

#include "gradientDescend.h" 
#include "inputSet.h"
#include "model.h"

GradientDescend::GradientDescend(Model& model, const InputSet& inputSet, unsigned int passes, double learnRate,double regularization, const unsigned int decimation):
    _model(model),
    _inputSet(inputSet),
    _passes(passes),
    _learnRate(learnRate),
    _regularization(regularization),
    _decimation(decimation),
    _decimationCount(0)
{}

GradientDescend::~GradientDescend() {
    
}

double GradientDescend::train() {
    auto start = std::chrono::high_resolution_clock::now();
    _decimationCount = 0;
    std::cout<<"trainSet total loss: " << _model.calcTotalLoss(_inputSet.validationSet())<<std::endl;
    
    for(unsigned int p = 0; p < _passes; ++p) {
        _pass();
        _printTrainResult(p);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    }
    std::cout<<"final total loss: " <<_model.calcTotalLoss(_inputSet.validationSet())<<std::endl;
    return _model.calcTotalLoss(_inputSet.validationSet());
}

void GradientDescend::_pass() {
    auto & batch = _inputSet.batch();
    _model.calcTotalLossGradient(batch);
    std::cout<<"*"<<std::flush;
    for( unsigned int ll = 0; ll < _model.getLayerCount(); ++ll) {
        Layer& l = _model.getLayer(ll);
        
        auto& biasSumGradient = l.biasSumGradient();
        unsigned int i = 0;
        for(auto& b: l.bias()){
            //std::cout<<" bias "<<biasSumGradient[i]<<std::endl;
            b -= biasSumGradient[i] * (_learnRate / batch.size());
            ++i;
        }
        std::cout<<"*"<<std::flush;;
        auto& weightSumGradient = l.weightSumGradient();
        i = 0;
        for(auto& w: l.weight()){
            //std::cout<<" weight "<<weightSumGradient[i]<<std::endl;
            w = (_regularization * w) - weightSumGradient[i] * (_learnRate / batch.size());
            ++i;
        }
        l.consolidateResult();
        std::cout<<"*"<<std::flush;
    }
    std::cout<<"*"<<std::endl;
    std::cout<<"intermediate loss "<< _model.calcTotalLoss(batch)<<std::endl;
}

void GradientDescend::_printTrainResult(const unsigned int pass) {
    if(++_decimationCount >= _decimation) {
        std::cout<<"pass: "<< pass + 1 <<"/"<<_passes<< " total loss: " <<_model.calcTotalLoss(_inputSet.validationSet())<<std::endl;
        _decimationCount = 0;
    }
}
