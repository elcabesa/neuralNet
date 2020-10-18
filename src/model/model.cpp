#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

#include "activation.h"
#include "labeledExample.h"
#include "inputSet.h"
#include "model.h"
#include "parallelDenseLayer.h"

Model::Model():_totalLoss{0}, _avgLoss{0} {}

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
    //std::cout<<"forward pass"<<std::endl;
    const Input* in = &input;
    if(verbose) { std::cout<<"*****************"<<std::endl;}
    for(auto& p: _layers) {
        p->propagate(*in);
        if(verbose) {
            p->printMinMax();
        }
        in = &p->output();
    }
    /*if(verbose) {
        in->print();
    }*/
    return *in;
}

double Model::calcLoss(const LabeledExample& le, bool verbose) {
    //std::cout<<"calcLoss"<<std::endl;
    auto& out = forwardPass(le.features());
    auto c = cost.calc(out.get(0), le.label());
    if (verbose) { std::cerr<< out.get(0) <<","<<le.label()<<","<<c<<std::endl;}
    return c;
}

double Model::calcAvgLoss(const std::vector<std::shared_ptr<LabeledExample>>& input, bool verbose, unsigned int count/* = 30*/) {
    //std::cout<<"calcAvgLoss"<<std::endl;
    double error = 0.0;
    unsigned int n = 0;
    //double avg = -0.271935;
    //double sum = 0.0;
    for(auto& le: input) {
        //sum += std::pow((*le).label() -avg, 2.0);
        error += calcLoss(*le, verbose);
        if(verbose && ++n >= count) {
            break;
        }
    }
    //std::cout<<"std dev:"<< sqrt(sum/input.size());
    return error / input.size();
}

double Model::getAvgLoss() const {
    return _avgLoss;
}

void Model::calcLossGradient(const LabeledExample& le) {
    //std::cout<<"calcLossGradient"<<std::endl;
    auto& out = forwardPass(le.features());
    _totalLoss += cost.calc(out.get(0), le.label());
    for (auto actualLayer = _layers.rbegin(); actualLayer!= _layers.rend(); ++actualLayer) {
        auto nextLayer = actualLayer + 1;
        
        std::vector<double> h;
        if(actualLayer == _layers.rbegin()) {
            h = {cost.derivate(out.get(0), le.label())};
        } else {
            h = (*(actualLayer - 1))->backPropHelper();  
        }
        const Input& input = (nextLayer != _layers.rend()? (*nextLayer)->output(): le.features());

        (*actualLayer)->backwardPropagate(h, input);
    }
}

void Model::calcTotalLossGradient(const std::vector<std::shared_ptr<LabeledExample>>& input) {
    for(auto& l :_layers) {
        (*l).resetSum();
    }
    _totalLoss = 0.0;
    for(auto& ex: input) {
        //std::cout<<"batch size "<<ex->features().getElementNumber()<<std::endl;
        calcLossGradient(*ex);
    }
    _avgLoss = _totalLoss / input.size();
}

#define VERSION "0002"
void Model::serialize(std::ofstream& ss) const {
    ss<<"{V:"<<VERSION<<"}";
    ss<<"{";
    for(auto& l :_layers) {
        l->serialize(ss);
    }
    ss<<"}"<<std::endl;
}

bool Model::deserialize(std::ifstream& ss) {
    //std::cout<<"DESERIALIZE MODEL"<<std::endl;
    if(ss.get() != '{') {std::cout<<"MODEL missing {"<<std::endl;return false;}
    if(ss.get() != 'V') {std::cout<<"MODEL missing V"<<std::endl;return false;}
    if(ss.get() != ':') {std::cout<<"MODEL missing :"<<std::endl;return false;}
    char buffer[4];
    ss.read(buffer, 4);
    std::string v(buffer, 4);
    if(v != VERSION) {std::cout<<"WRONG NETWORK VERSION: "<<v<<" expected: "<<VERSION<<std::endl;return false;}
    if(ss.get() != '}') {std::cout<<"MODEL missing }"<<std::endl;return false;}
    if(ss.get() != '{') {std::cout<<"MODEL missing {"<<std::endl;return false;}
    for(auto& l :_layers) {
        if(!l->deserialize(ss)) {std::cout<<"MODEL internal layer error"<<std::endl;return false;}
    }
    if(ss.get() != '}') {std::cout<<"MODEL missing }"<<std::endl;return false;} 
    return true;
}

Layer& Model::getLayer(unsigned int index) {
    assert(index < getLayerCount());
    return *(_layers[index]);
}
unsigned int Model::getLayerCount() {
    return _layers.size();
}

void Model::printParamsStats() {
/*    std::cout<<"layer count = " << _layers.size()<<std::endl;
    std::cout<<"---------------------------------"<<std::endl;
    for(auto &l: _layers) {
        std::cout<<(*l).getInputSize() <<" x "<<(*l).getOutputSize()<<std::endl;
    }
    std::cout<<"---------------------------------"<<std::endl;
    for(auto &l: _layers) {
        std::cout<<"biases: "<<(*l).bias().size() <<" weight: "<<(*l).weight().size() <<std::endl;
    }
    std::cout<<"---------------------------------"<<std::endl;
    unsigned int count = 0;
    for(auto &l: _layers) {
        std::cout<<"LAYER "<<(++count)<<std::endl;
        double biasAvg = std::accumulate( (*l).bias().begin(), (*l).bias().end(), 0.0) / (*l).bias().size(); 
        std::cout<<"bias avg "<< biasAvg<<std::endl;
        double biasMin = *std::min_element( (*l).bias().begin(), (*l).bias().end()); 
        std::cout<<"bias min "<< biasMin<<std::endl;
        double biasMax = *std::max_element( (*l).bias().begin(), (*l).bias().end()); 
        std::cout<<"bias max "<< biasMax<<std::endl;
        
        double weightAvg = std::accumulate( (*l).weight().begin(), (*l).weight().end(), 0.0) / (*l).weight().size(); 
        std::cout<<"weight avg "<< weightAvg<<std::endl;
        double weightMin = *std::min_element( (*l).weight().begin(), (*l).weight().end()); 
        std::cout<<"weight min "<< weightMin<<std::endl;
        double weightMax = *std::max_element( (*l).weight().begin(), (*l).weight().end()); 
        std::cout<<"weight max "<< weightMax<<std::endl;
    }*/
}

void Model::clear() {
    _layers.clear();
}

void Model::VerifyTotalLossGradient(const std::vector<std::shared_ptr<LabeledExample>>& input) {
    const double delta = 0.00001;
    const double maxError = 1e-6;
    //std::cout<<"eccomi"<<std::endl;
    for(unsigned int l = 0; l < getLayerCount(); ++l) {
        auto& actualLayer = getLayer(l);
        //std::cout<<"layer "<<l<<std::endl;
        
        for(unsigned int i = 0; auto& b : actualLayer.bias()) {
            //std::cout<<"\tbias "<<i<<std::endl;
            std::cout<<"\rlayer "<<l<<" bias "<<i<<"\t\t";
            double grad = 0.0;
            for(auto& e :input) {
                auto originalB = b;
                b = originalB + delta;
                auto lplus = calcLoss(*(e));
                b = originalB - delta;
                auto lminus = calcLoss(*(e));
                b = originalB;
                grad += (lplus - lminus)/(2.0 * delta);
            }
            if(std::abs(actualLayer.getBiasSumGradient(i) - grad) > maxError) {
                std::cout<<"EEEEEEEERRRRRRORE"<<std::endl;
                exit(-1);
            }
            ++i;
        }

        for(unsigned int i = 0; auto& w : actualLayer.weight()) {
            //std::cout<<"\tweight "<<i<<std::endl;
            std::cout<<"\rlayer "<<l<<" weight "<<i<<"\t\t";
            double grad = 0.0;
            for(auto& e :input) {
                auto originalW = w;
                w = originalW + delta;
                auto lplus = calcLoss(*(e));
                w = originalW - delta;
                auto lminus = calcLoss(*(e));
                w = originalW;
                grad += (lplus - lminus)/(2.0 * delta);
            }
            if(std::abs(actualLayer.getWeightSumGradient(i) - grad) > maxError) {
                std::cout<<"EEEEEEEERRRRRRORE"<<std::endl;
                exit(-1);
            }
            ++i;
        }
    }
    std::cout<<std::endl;
}

void Model::setQuantization(bool q) {
    for( auto& l: _layers) {
        l->setQuantization(q);
    }
}