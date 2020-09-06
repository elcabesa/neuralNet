#ifndef _PARALLEL_DENSE_LAYER_H
#define _PARALLEL_DENSE_LAYER_H

#include <memory>
#include <vector>

#include "layer.h"
#include "denseLayer.h"

class Activation;

class ParallelDenseLayer: public Layer {
public:
    ParallelDenseLayer(const unsigned int number, const unsigned int inputSize, const unsigned int outputSize, std::shared_ptr<Activation> act, const double stdDev = 0.0);
    ~ParallelDenseLayer();
    
    void propagate(const Input& input);
    void printParams() const;
    void randomizeParams();
    void backwardCalcBias(const std::vector<double>& h);
    void backwardCalcWeight(const Input& prevOut);
    std::vector<double> backPropHelper() const;
    
    void resetSum();
    void accumulateGradients();
    
    std::vector<double>& bias();
    std::vector<double>& weight();
    
    void consolidateResult();
    
    std::vector<double>& biasSumGradient();
    std::vector<double>& weightSumGradient();
    
    unsigned int _calcWeightIndex(const unsigned int layer, const unsigned int i, const unsigned int o) const;
    unsigned int _calcBiasIndex(const unsigned int layer, const unsigned int o) const;
    
    void serialize(std::ofstream& ss) const;
    bool deserialize(std::ifstream& ss);
    
private:
    std::vector<double> _bias;
    std::vector<double> _weight;
    std::vector<double> _biasSumGradient;
    std::vector<double> _weightSumGradient;
    std::vector<DenseLayer> _parallelLayers;
    const unsigned int _number;

};

#endif  
