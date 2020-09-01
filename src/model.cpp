
#include <iostream>
#include "activation.h"
#include "labeledExample.h"
#include "model.h"

void Model::addLayer(std::unique_ptr<Layer> l) {
    if(_layers.size() > 0 && _layers.back()->getOutputSize() != l->getInputSize()) {
        std::cerr<<"ERROR in SIZING"<< std::endl;
        exit(0);
    }
        
    _layers.push_back(std::move(l));
}

void Model::randomizeParams() {
    for(auto& p: _layers) {p->randomizeParams();}
}

void Model::printParams() {
    for(auto& p: _layers) {p->printParams();}
}

const std::vector<double>& Model::forwardPass(std::vector<double> input, bool verbose /* = false */) {
    const std::vector<double>* in = &input;
    for(auto& p: _layers) {
        if(verbose){p->printOutput();}
        p->propagate(*in);
        in = &p->output();
    }
    if(verbose){for(auto& el: *in) {std::cout<< el<< " ";} std::cout<<std::endl;}
    return *in;
}

double Model::calcLoss(const LabeledExample& le) {
    auto& out = forwardPass(le.features);
    return cost.calc(out[0], le.label[0]);
}

double Model::calcTotalLoss(const std::vector<LabeledExample>& input) {
    double error = 0.0;
    for(auto& le: input) {error += calcLoss(le);}
    return error / input.size();
}

void Model::calcLossGradient(const LabeledExample& le) {
    auto& out = forwardPass(le.features);
    for (auto actualLayer = _layers.rbegin(); actualLayer!= _layers.rend(); ++actualLayer) {
        
        // todo uniformare
        if(actualLayer == _layers.rbegin()) {
            // todo manage multi output network
            std::vector<double> h = {cost.derivate(out[0], le.label[0])};
            (*actualLayer)->backwardCalcBias(h);
        }
        else {
            auto PreviousLayer = actualLayer - 1;
            auto h = (*PreviousLayer)->backPropHelper();
            (*actualLayer)->backwardCalcBias(h);
        }
        
        auto nextLayer = actualLayer + 1;
        if(nextLayer != _layers.rend()) {
            const std::vector<double>& PreviousOut = (*nextLayer)->output();
            (*actualLayer)->backwardCalcWeight(PreviousOut);
        }
        else {
            const std::vector<double>& PreviousOut = le.features;
            (*actualLayer)->backwardCalcWeight(PreviousOut);
        }
        
    }
}

void Model::calcTotalLossGradient(const std::vector<LabeledExample>& input) {
    for(auto& l :_layers) {
        (*l).resetSum();
    }
    for(auto& ex: input) {
        calcLossGradient(ex);
        for(auto& l :_layers) {
            (*l).accumulateGradients();
        }
    }
}

double Model::train(unsigned int passes, double learnRate, const std::vector<LabeledExample>& trainSet, const std::vector<LabeledExample>& validationSet, double regularization, const unsigned int decimation) {
    unsigned int decimationCount = 0;
    std::cout<<"trainSet total loss: " << calcTotalLoss(trainSet)<<std::endl;
    for(unsigned int p = 0; p < passes; ++p) {
        calcTotalLossGradient(trainSet);
        for(auto& l :_layers) {
            auto& biasSumGradient = l->biasSumGradient();
            unsigned int i = 0;
            for(auto& b: l->bias()){
                //std::cout<<" bias "<<biasSumGradient[i]<<std::endl;
                b -= biasSumGradient[i] * learnRate / trainSet.size();
                ++i;
            }
            auto& weightSumGradient = l->weightSumGradient();
            i = 0;
            for(auto& w: l->weight()){
                //std::cout<<" weight "<<weightSumGradient[i]<<std::endl;
                w = (regularization * w) - weightSumGradient[i] * learnRate / trainSet.size();
                ++i;
            }
        }
        if(++decimationCount >= decimation) {
            std::cout<<"pass: "<< p + 1 <<"/"<<passes<< " total loss: " <<calcTotalLoss(trainSet)<<" "<<calcTotalLoss(validationSet)<<std::endl;
            decimationCount = 0;
        }
    }
    std::cout<<"final total loss: " <<calcTotalLoss(trainSet)<<" "<<calcTotalLoss(validationSet)<<std::endl;
    return calcTotalLoss(validationSet);
    
}
