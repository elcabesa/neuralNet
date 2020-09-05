#include <iostream>

#include "activation.h"
#include "labeledExample.h"
#include "inputSet.h"
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

const Input& Model::forwardPass(const Input& input, bool verbose /* = false */) {
    const Input* in = &input;
    for(auto& p: _layers) {
        if(verbose){p->printOutput();}
        p->propagate(*in);
        in = &p->output();
    }
    if(verbose){in->print();}
    return *in;
}

double Model::calcLoss(const LabeledExample& le) {
    auto& out = forwardPass(le.features());
    return cost.calc(out.get(0), le.label());
}

double Model::calcTotalLoss(const std::vector<std::shared_ptr<LabeledExample>>& input) {
    double error = 0.0;
    for(auto& le: input) {error += calcLoss(*le);}
    return error / input.size();
}

void Model::calcLossGradient(const LabeledExample& le) {
    auto& out = forwardPass(le.features());
    for (auto actualLayer = _layers.rbegin(); actualLayer!= _layers.rend(); ++actualLayer) {
        
        // todo uniformare
        if(actualLayer == _layers.rbegin()) {
            // todo manage multi output network
            std::vector<double> h = {cost.derivate(out.get(0), le.label())};
            (*actualLayer)->backwardCalcBias(h);
        }
        else {
            auto PreviousLayer = actualLayer - 1;
            auto h = (*PreviousLayer)->backPropHelper();
            (*actualLayer)->backwardCalcBias(h);
        }
        
        auto nextLayer = actualLayer + 1;
        if(nextLayer != _layers.rend()) {
            const Input& PreviousOut = (*nextLayer)->output();
            (*actualLayer)->backwardCalcWeight(PreviousOut);
        }
        else {
            const Input& PreviousOut = le.features();
            (*actualLayer)->backwardCalcWeight(PreviousOut);
        }
        (*actualLayer)->accumulateGradients();
        
    }
}

void Model::calcTotalLossGradient(const std::vector<std::shared_ptr<LabeledExample>>& input) {
    for(auto& l :_layers) {
        (*l).resetSum();
    }
    unsigned int count = 0;
    for(auto& ex: input) {
        calcLossGradient(*ex);
        if((input.size() > 20) && (count % (input.size()/20)) ==0) {
            std::cout<<"+"<<std::flush;
        }
        ++count;
    }
}


void Model::serialize(std::ofstream& ss) const {
    ss<<"{";
    for(auto& l :_layers) {
        l->serialize(ss);
    }
    ss<<"}"<<std::endl;
}

bool Model::deserialize(std::ifstream& ss) {
    //std::cout<<"DESERIALIZE MODEL"<<std::endl;
    if(ss.get() != '{') {std::cout<<"MODEL missing {"<<std::endl;return false;}
    for(auto& l :_layers) {
        if(!l->deserialize(ss)) {std::cout<<"MODEL internal layer error"<<std::endl;return false;}
    }
    if(ss.get() != '}') {std::cout<<"MODEL missing }"<<std::endl;return false;} 
    return true;
}

Layer& Model::getLayer(unsigned int index) {
    return *(_layers[index]);
}
unsigned int Model::getLayerCount() {
    return _layers.size();
}
