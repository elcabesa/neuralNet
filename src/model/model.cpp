#include <cassert>
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
    const Input* in = &input;
    for(auto& p: _layers) {
        p->propagate(*in);
        if(verbose) {
            p->printOutput();
        }
        in = &p->output();
    }
    if(verbose) {
        in->print();
    }
    return *in;
}

double Model::calcLoss(const LabeledExample& le, bool verbose) {
    auto& out = forwardPass(le.features());
    auto c = cost.calc(out.get(0), le.label());
    if (verbose) { std::cerr<< out.get(0) <<","<<le.label()<<","<<c<<std::endl;}
    return c;
}

double Model::calcAvgLoss(const std::vector<std::shared_ptr<LabeledExample>>& input, bool verbose, unsigned int count/* = 30*/) {
    double error = 0.0;
    unsigned int n = 0;
    for(auto& le: input) {
        error += calcLoss(*le, verbose);
        if(verbose && ++n >= count) {
            break;
        }
    }
    return error / input.size();
}

double Model::getAvgLoss() const {
    return _avgLoss;
}

void Model::calcLossGradient(const LabeledExample& le) {
    auto& out = forwardPass(le.features());
    _totalLoss += cost.calc(out.get(0), le.label());
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
            const Input& input = (*nextLayer)->output();
            (*actualLayer)->backwardCalcWeight(input);
            (*actualLayer)->accumulateGradients(input);
        }
        else {
            const Input& input = le.features();
            (*actualLayer)->backwardCalcWeight(input);
            (*actualLayer)->accumulateGradients(input);
        }
        
    }
}

void Model::calcTotalLossGradient(const std::vector<std::shared_ptr<LabeledExample>>& input) {
    for(auto& l :_layers) {
        (*l).resetSum();
    }
    _totalLoss = 0.0;
    for(auto& ex: input) {
        calcLossGradient(*ex);
    }
    _avgLoss = _totalLoss / input.size();
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
        ParallelDenseLayer*  pdl = dynamic_cast<ParallelDenseLayer*>(&actualLayer);
        if(!pdl) {
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
        } else {
            for(unsigned int n = 0; n < pdl->getLayerNumber(); ++n) {
                //std::cout<<"\tparallel layer "<<n<<std::endl;
                auto & layer = pdl->getLayer(n);
                for(unsigned int i = 0; auto& b : layer.bias()) {
                    //std::cout<<"\t\tbias "<<i<<std::endl;
                    std::cout<<"\rlayer "<<l<<" subLayer "<<n<<" bias "<<i<<"\t\t";
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
                    if(std::abs(layer.getBiasSumGradient(i) - grad) > maxError) {
                        std::cout<<"EEEEEEEERRRRRRORE"<<std::endl;
                        exit(-1);
                    }
                    ++i;
                }

                for(unsigned int i = 0; auto& w : layer.weight()) {
                    //std::cout<<"\t\tweight "<<i<<std::endl;
                    std::cout<<"\rlayer "<<l<<" subLayer "<<n<<" weight "<<i<<"\t\t";
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
                    if(std::abs(layer.getWeightSumGradient(i) - grad) > maxError) {
                        std::cout<<"EEEEEEEERRRRRRORE"<<std::endl;
                        exit(-1);
                    }
                    ++i;
                }
            }
        }
    }
    std::cout<<std::endl;
}