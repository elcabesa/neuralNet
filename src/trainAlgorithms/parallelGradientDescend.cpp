#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "parallelGradientDescend.h" 
#include "inputSet.h"
#include "model.h"

ParallelGradientDescend::ParallelGradientDescend(Model& model, const InputSet& inputSet, unsigned int passes, double learnRate,double regularization, double beta):
    _model(model),
    _inputSet(inputSet),
    _passes(passes),
    _learnRate(learnRate),
    _regularization(regularization),
    _counter(0),
    _beta(beta),
    _min(1e20)
{
    for(unsigned int ll = 0; ll< model.getLayerCount(); ++ll) {
        auto& l= model.getLayer(ll);
        std::vector<double> bias(l.bias().size(), 0);
        _v_b.push_back(bias);
        
        std::vector<double> weight(l.weight().size(), 0);
        _v_w.push_back(weight);
    }
    
}

ParallelGradientDescend::~ParallelGradientDescend() {
    
}

double ParallelGradientDescend::train() {
    std::cerr <<"TrainsetError,ValidationError"<<std::endl;
    //auto start = std::chrono::high_resolution_clock::now();
    
    std::cout<<"initiali ValidationSet avg loss: " << _model.calcAvgLoss(_inputSet.validationSet())<<std::endl;
    
    for(unsigned int p = 0; p < _passes; ++p) {
        _pass();
        _save(p);
        //_printTrainResult(p);
    }
    std::cout<<"final ValidationSet avg loss: " <<_model.calcAvgLoss(_inputSet.validationSet())<<std::endl;
    return _model.calcAvgLoss(_inputSet.validationSet());
}

void ParallelGradientDescend::_pass() {
    _calcLossGradient();
    _updateParams();
    std::cout<<"intermediate loss "<< _model.getAvgLoss() <<std::endl;
    std::cerr <<sqrt(_model.getAvgLoss())<<","<<std::endl;
}

void ParallelGradientDescend::_calcLossGradient() {
    _model.calcTotalLossGradient(_inputSet.batch());
}

void ParallelGradientDescend::_updateParams() {
    for( unsigned int ll = 0; ll < _model.getLayerCount(); ++ll) {
        Layer& l = _model.getLayer(ll);
        auto& _v_b_l = _v_b[ll];
        unsigned int i = 0;
        for(auto& b: l.bias()){
            double gradBias = l.getBiasSumGradient(i);
            _v_b_l[i] = _beta * _v_b_l[i] + (1 - _beta) * gradBias * gradBias;
            b -= gradBias * (_learnRate / sqrt(_v_b_l[i] + 1e-8));
            ++i;
        }
        
        auto& _v_w_l = _v_w[ll];
        i = 0;
        for(auto& w: l.weight()){
            double gradWeight = l.getWeightSumGradient(i);
            _v_w_l[i] = _beta * _v_w_l[i] + (1 - _beta) * gradWeight * gradWeight;
            w = (_regularization * w) - gradWeight * (_learnRate / sqrt(_v_w_l[i] + 1e-8));
            ++i;
        }
        l.consolidateResult();

    }
}


void ParallelGradientDescend::_printTrainResult(const unsigned int pass) {
    //auto finish = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> elapsed = finish - start;
    //std::cout << "Elapsed time: " << elapsed.count() << " s\n";    
    double l = _model.calcAvgLoss(_inputSet.validationSet());
    std::cout<<"pass: "<< pass + 1 <<"/"<<_passes<< " total loss: " << l << " minloss "<<_min<<std::endl;
    std::cerr <<sqrt(l)<<std::endl;
    
    
}


void ParallelGradientDescend::_save(const unsigned int pass) {
    bool save = false;
    _counter++;
    if(_counter>=1000) {
        save = true;
        _counter = 0;
    }
    
    double l = _model.getAvgLoss();
    if(l<_min ) {
        _min = l;
        save = true;
    }
    if(save) {
        std::ofstream nnFile;
        std::string fileName;
        fileName += "nn-";
        fileName += std::to_string(pass);
        fileName += "-";
        fileName += std::to_string(sqrt(l));
        fileName += ".txt";
        nnFile.open (fileName);
        _model.serialize(nnFile);
        nnFile.close();
    }
}
