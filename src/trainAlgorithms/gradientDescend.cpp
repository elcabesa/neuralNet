#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

#include "gradientDescend.h" 
#include "labeledExample.h"
#include "inputSet.h"
#include "model.h"

GradientDescend::GradientDescend(Model& model, const InputSet& inputSet, unsigned int passes, double learnRate,double regularization, double beta):
    _model(model),
    _inputSet(inputSet),
    _passes(passes),
    _learnRate(learnRate),
    _regularization(regularization),
    _beta(beta),
    _min(1e20),
    _accumulatorLoss(0),
    _count(0)
{}

GradientDescend::~GradientDescend() {
    
}

double GradientDescend::train() {
    std::cerr <<"TrainsetError,ValidationError"<<std::endl;
    //auto start = std::chrono::high_resolution_clock::now();
    bool infinite = (0 == _passes);
    std::cout<<"initiali ValidationSet avg loss: " << sqrt(_model.calcAvgLoss(_inputSet.validationSet()))<<std::endl;
    
    for(unsigned int p = 0; infinite || p < _passes; ++p) {
        _pass();
        _printTrainResult(p);
    }
    std::cout<<"final ValidationSet avg loss: " <<sqrt(_model.calcAvgLoss(_inputSet.validationSet()))<<std::endl;
    return _model.calcAvgLoss(_inputSet.validationSet());
}

void GradientDescend::_pass() {
    auto & batch = _inputSet.batch();
    _model.calcTotalLossGradient(batch);
    //std::cout<<"-----------------"<<std::endl;
    for( unsigned int ll = 0; ll < _model.getLayerCount(); ++ll) {
        Layer& l = _model.getLayer(ll);
        l.upgradeBias(_beta, _learnRate);
        l.upgradeWeight(_beta, _learnRate, _regularization);
    }
    //std::cout<<"intermediate loss "<< sqrt(_model.getAvgLoss()) <<std::endl;
    _accumulatorLoss += _model.getAvgLoss();
    ++_count;
}

void GradientDescend::_printTrainResult(const unsigned int pass) {
    //auto finish = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> elapsed = finish - start;
    //std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    
    if(/*l<_min*/(pass%50000)==0) {
       // _min = l;
        double l = /*_model.getAvgLoss();*/_model.calcAvgLoss(_inputSet.validationSet());
        unsigned int p = pass /50000;
        std::cout<<"pass: "<< p<<" loss "<<sqrt(l) <<std::endl;
        std::ofstream nnFile;
        std::string fileName;
        fileName += "nn-";
        fileName += std::to_string(p);
        //fileName += "-";
        //fileName += std::to_string(sqrt(l));
        fileName += ".txt";
        nnFile.open (fileName);
        _model.serialize(nnFile);
        nnFile.close();
        std::cerr <<sqrt(_accumulatorLoss/_count)<<",";
        std::cerr <<sqrt(l)<<std::endl;
        _accumulatorLoss = 0;
        _count = 0;
    }
    //std::cout<<"pass: "<< pass + 1 <<"/"<<_passes<< " total loss: " << l << " minloss "<<_min<<std::endl;
    
    
    
}
