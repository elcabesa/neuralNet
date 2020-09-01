#ifndef _DENSE_LAYER_H
#define _DENSE_LAYER_H

#include <memory>
#include <vector>

#include "layer.h"

class Activation;

class DenseLayer: public Layer {
public:
    DenseLayer(const unsigned int inputSize, const unsigned int outputSize, std::unique_ptr<Activation> act);
    ~DenseLayer();

    //std::vector<double>& biasGradient();
    //std::vector<double>& weightGradient();
    //const std::vector<double>& netOutput() const;
    //const Activation& activation() const;
    
    void propagate(const std::vector<double>& input);
    void printParams() const;
    void randomizeParams();
    void backwardCalcBias(const std::vector<double>& h);
    void backwardCalcWeight(const std::vector<double>& prevOut);
    std::vector<double> backPropHelper() const;
    
    void resetSum();
    void accumulateGradients();
    
    std::vector<double>& bias();
    std::vector<double>& weight();
    
    std::vector<double>& biasSumGradient();
    std::vector<double>& weightSumGradient();
    
private:
    std::vector<double> _bias;
    std::vector<double> _weight;
    std::vector<double> _biasGradient;
    std::vector<double> _weightGradient;
    std::vector<double> _biasSumGradient;
    std::vector<double> _weightSumGradient;
    std::vector<double> _netOutput;
    std::unique_ptr<Activation> _act;
    
    unsigned int _calcWeightIndex(const unsigned int i, const unsigned int o) const;
    void calcNetOut(const std::vector<double>& input);
    void calcOut();
};

#endif  
