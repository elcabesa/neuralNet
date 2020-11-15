#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

#include "gradientDescend.h" 
#include "labeledExample.h"
#include "inputSet.h"
#include "model.h"
#include "pedoneCheck.h"

GradientDescend::GradientDescend(Model& model, const InputSet& inputSet, unsigned int passes, double learnRate,double regularization, double beta, unsigned int quant, bool rmsprop):
    _model(model),
    _inputSet(inputSet),
    _passes(passes),
    _learnRate(learnRate),
    _regularization(regularization),
    _beta(beta),
    _min(1e20),
    _PedoneAccumulatorLoss(0.0),
    _VajoletAccumulatorLoss(0.0),
    _count(0),
    _quantization(false),
    _quantizationPass(quant),
    _rmsProp(rmsprop)/*,
    _totalLoss(0.0),
    _totalCount(0)*/
{}

GradientDescend::~GradientDescend() {
    
}

double GradientDescend::train(unsigned int decimation) {
    std::cerr <<"TrainsetError,ValidationError"<<std::endl;
    //auto start = std::chrono::high_resolution_clock::now();
    bool infinite = (0 == _passes);
    _model.setQuantization(true);
    double avgLoss = _model.calcAvgLoss(_inputSet.validationSet());
    std::cout<<"initial ValidationSet avg loss: " << sqrt(2.0 * avgLoss) / 10000.0 <<std::endl;
    _model.setQuantization(_quantization);
    
    if(_pedone) {delete _pedone;}
    _pedone = new PedoneCheck(&_model);
    _pedone->caricaPesi();

    for(unsigned int p = 1; infinite || p <= _passes; ++p) {
        if (_quantizationPass != 0 && p >= _quantizationPass) {_quantization = true;}
        else {_quantization = false;}
        _model.setQuantization(_quantization);
        _pass(p);
        _printTrainResult(p, decimation);
    }
    /*_model.setQuantization(true);
    double r = _model.calcAvgLoss(_inputSet.validationSet());
    std::cout<<"final ValidationSet avg loss: " <<sqrt(2.0 * r) / 10000.0 <<std::endl;
    _model.setQuantization(false);   */ 
    return 0;
}

void GradientDescend::_pass(const unsigned int pass) {
    //std::cout<<"GRADIENT DESCENT PASS "<< pass<<std::endl;
    auto & batch = _inputSet.batch();
    //std::cout<<"batchsize "<<batch.size()<<std::endl;
    _model.calcTotalLossGradient(batch);
    // TODO REMOVE
    //_pedone->caricaPesi();
    _pedone->caricaInput((*batch[0]));
    _pedone->calcolaRisRete();
    double pedLoss = _model.getOutputScaling() * _model.getOutputScaling() * _pedone->calcGrad((*batch[0]).label() / _model.getOutputScaling());
    //_model.VerifyTotalLossGradient(batch);
    
    //std::cout<<"-----------------"<<std::endl;
    for( unsigned int ll = 0; ll < _model.getLayerCount(); ++ll) {
        //std::cout<<"layer "<<ll<<std::endl;
        Layer& l = _model.getLayer(ll);
        l.upgradeBias(_beta, _learnRate, _rmsProp);
        l.upgradeWeight(_beta, _learnRate, _regularization, _rmsProp);
    }
    _pedone->updateweights(_learnRate);
    //double avgLoss = _model.calcAvgLoss(batch);
    //std::cout<<sqrt(_model.getAvgLoss())<<" "<< sqrt(avgLoss) <<std::endl;
    //std::cout<<"intermediate loss "<< sqrt(_model.getAvgLoss()) <<std::endl;
    /*if(_model.getAvgLoss() > 400 ){
        std::cout<<"WARNING AVG LOSS "<<_model.getAvgLoss()<<std::endl;
    }*/
    _PedoneAccumulatorLoss += pedLoss;
    _VajoletAccumulatorLoss += _model.getAvgLoss();
    ++_count;
}

void GradientDescend::_printTrainResult(const unsigned int pass, unsigned int decimation) {
    if((pass%decimation)==0) 
    {
        unsigned int p = pass / decimation;
        /*_model.setQuantization(true);
        double l = _model.calcAvgLoss(_inputSet.validationSet());
        _model.setQuantization(_quantization);*/
        double pl = _PedoneAccumulatorLoss/_count;
        double vl = _VajoletAccumulatorLoss/_count;
        //std::cout << "pass: " << p << " loss "<< sqrt(_accumulatorLoss/_count)<<std::endl;
        

        std::cerr <<sqrt(2.0 * pl) / 10000.0 <<",";
        //_totalLoss += _accumulatorLoss;
        //_totalCount += _count;
        std::cerr <<sqrt(2.0 * vl) / 10000.0 <<std::endl;

        std::cout << "pass: " << p << " loss "<< sqrt(2.0 * pl) / 10000.0 << " " <<sqrt(2.0 * vl) / 10000.0<<std::endl;
        //std::cout << sqrt(_totalLoss/_totalCount)<<std::endl;
        //_save(p);

        
        //std::cout<<" AVGLOSS " << _lossLowPassFilter<<std::endl; 
        _PedoneAccumulatorLoss = 0.0;
        _VajoletAccumulatorLoss = 0.0;
        _count = 0;
    }
}

void GradientDescend::_save(const unsigned int pass) {
    std::ofstream nnFile;
    std::string fileName;
    fileName += "save/nn-";
    fileName += std::to_string(pass);
    fileName += ".txt";
    nnFile.open (fileName);
    _model.serialize(nnFile);
    nnFile.close();
}
