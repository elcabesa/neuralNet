#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "gradientDescend.h" 
#include "inputSet.h"
#include "model.h"

GradientDescend::GradientDescend(Model& model, const InputSet& inputSet, unsigned int passes, double learnRate,double regularization):
    _model(model),
    _inputSet(inputSet),
    _passes(passes),
    _learnRate(learnRate),
    _regularization(regularization),
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

GradientDescend::~GradientDescend() {
    
}

double GradientDescend::train() {
    std::cerr <<"TrainsetError,ValidationError"<<std::endl;
    //auto start = std::chrono::high_resolution_clock::now();
    
    std::cout<<"initiali ValidationSet avg loss: " << _model.calcAvgLoss(_inputSet.validationSet())<<std::endl;
    
    for(unsigned int p = 0; p < _passes; ++p) {
        _pass();
        _printTrainResult(p);
    }
    std::cout<<"final ValidationSet avg loss: " <<_model.calcAvgLoss(_inputSet.validationSet())<<std::endl;
    return _model.calcAvgLoss(_inputSet.validationSet());
}

void GradientDescend::_pass() {
    auto & batch = _inputSet.batch();
    _model.calcTotalLossGradient(batch);
    
    for( unsigned int ll = 0; ll < _model.getLayerCount(); ++ll) {
        Layer& l = _model.getLayer(ll);
        auto& _v_b_l = _v_b[ll];
        const double beta = 0.9;
        
        unsigned int i = 0;
        for(auto& b: l.bias()){
            double gradBias = l.getBiasSumGradient(i);
            _v_b_l[i] = beta * _v_b_l[i] + (1-beta) * gradBias * gradBias;
            b -= gradBias * (_learnRate / sqrt(_v_b_l[i] + 1e-8));
            ++i;
        }
        
        auto& _v_w_l = _v_w[ll];
        i = 0;
        for(auto& w: l.weight()){
            double gradWeight = l.getWeightSumGradient(i);
            _v_w_l[i] = beta * _v_w_l[i] + (1-beta) * gradWeight * gradWeight;
            w = (_regularization * w) - gradWeight * (_learnRate / sqrt(_v_w_l[i] + 1e-8));
            ++i;
        }
        l.consolidateResult();

    }
    std::cout<<"intermediate loss "<< _model.getAvgLoss() <<std::endl;
    std::cerr <<sqrt(_model.getAvgLoss())<<",";
}

void GradientDescend::_printTrainResult(const unsigned int pass) {
    //auto finish = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> elapsed = finish - start;
    //std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    double l = _model.calcAvgLoss(_inputSet.validationSet());
    if(l<_min) {
        _min = l;
        
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
    std::cout<<"pass: "<< pass + 1 <<"/"<<_passes<< " total loss: " << l << " minloss "<<_min<<std::endl;
    std::cerr <<sqrt(l)<<std::endl;
    
    
}
