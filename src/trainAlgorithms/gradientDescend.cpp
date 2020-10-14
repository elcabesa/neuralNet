#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

#include "gradientDescend.h" 
#include "labeledExample.h"
#include "inputSet.h"
#include "model.h"

GradientDescend::GradientDescend(Model& model, const InputSet& inputSet, unsigned int passes, double learnRate,double regularization, double beta, unsigned int quant, bool rmsprop):
    _model(model),
    _inputSet(inputSet),
    _passes(passes),
    _learnRate(learnRate),
    _regularization(regularization),
    _beta(beta),
    _min(1e20),
    _accumulatorLoss(0),
    _count(0),
    _quantization(false),
    _quantizationPass(quant),
    _rmsProp(rmsprop)
{}

GradientDescend::~GradientDescend() {
    
}

double GradientDescend::train(unsigned int decimation) {
    std::cerr <<"TrainsetError,ValidationError"<<std::endl;
    //auto start = std::chrono::high_resolution_clock::now();
    bool infinite = (0 == _passes);
    _model.setQuantization(true);
    std::cout<<"initial ValidationSet avg loss: " << sqrt(_model.calcAvgLoss(_inputSet.validationSet()))<<std::endl;
    _model.setQuantization(_quantization);
    for(unsigned int p = 0; infinite || p < _passes; ++p) {
        if (_quantizationPass && p > _quantizationPass) {_quantization = true;}
        else {_quantization = false;}
        _model.setQuantization(_quantization);
        _pass(p);
        _printTrainResult(p, decimation);
    }
    _model.setQuantization(true);
    double r = _model.calcAvgLoss(_inputSet.validationSet());
    std::cout<<"final ValidationSet avg loss: " <<sqrt(r)<<std::endl;
    _model.setQuantization(false);    
    return r;
}

void GradientDescend::_pass(const unsigned int pass) {
    //std::cout<<"GRADIENT DESCENT PASS"<<std::endl;
    auto & batch = _inputSet.batch();
    //std::cout<<"batchsize "<<batch.size()<<std::endl;
    _model.calcTotalLossGradient(batch);
    //_model.VerifyTotalLossGradient(batch);
    
    //std::cout<<"-----------------"<<std::endl;
    for( unsigned int ll = 0; ll < _model.getLayerCount(); ++ll) {
        Layer& l = _model.getLayer(ll);
        l.upgradeBias(_beta, _learnRate, _rmsProp);
        l.upgradeWeight(_beta, _learnRate, _regularization, _rmsProp);
    }
    //double avgLoss = _model.calcAvgLoss(batch);
    //std::cout<<sqrt(_model.getAvgLoss())<<" "<< sqrt(avgLoss) <<std::endl;
    //std::cout<<"intermediate loss "<< sqrt(_model.getAvgLoss()) <<std::endl;
    /*if(_model.getAvgLoss() > 400 ){
        std::cout<<"WARNING AVG LOSS "<<_model.getAvgLoss()<<std::endl;
    }*/
    _accumulatorLoss += _model.getAvgLoss();
    ++_count;
}

void GradientDescend::_printTrainResult(const unsigned int pass, unsigned int decimation) {
    if((pass%decimation)==0) 
    {
        unsigned int p = pass / decimation;
        _model.setQuantization(true);
        double l = _model.calcAvgLoss(_inputSet.validationSet());
        _model.setQuantization(_quantization);

        std::cout << "pass: " << p << " loss "<< sqrt(l) << std::endl;
        _save(p);

        std::cerr <<sqrt(_accumulatorLoss/_count)<<",";
        std::cerr <<sqrt(l)<<std::endl;

        _accumulatorLoss = 0;
        _count = 0;
    }
}

void GradientDescend::_save(const unsigned int pass) {
    std::ofstream nnFile;
    std::string fileName;
    fileName += "nn-";
    fileName += std::to_string(pass);
    fileName += ".txt";
    nnFile.open (fileName);
    _model.serialize(nnFile);
    nnFile.close();
}
