#ifndef _PARALLEL_DENSE_LAYER_H
#define _PARALLEL_DENSE_LAYER_H

#include <memory>
#include <vector>

#include "layer.h"
#include "denseLayer.h"

class Activation;

class ParallelDenseLayer: public Layer {
public:
    ParallelDenseLayer(const unsigned int number, const unsigned int inputSize, const unsigned int outputSize, std::shared_ptr<Activation> act, unsigned int outScale, unsigned int weightScale, const unsigned int accumulatorBits, const double stdDev = 0.0);
    ~ParallelDenseLayer();
    
    void propagate(const Input& input);
    void printParams() const;
    void randomizeParams();
    void backwardCalcBiasGradient(const std::vector<double>& h);
    void backwardCalcWeightGradient(const Input& input);
    std::vector<double> backPropHelper() const;
    
    void resetSum();
    void accumulateGradients(const Input& input);
    
    std::vector<double>& bias();
    std::vector<double>& weight();
    
    DenseLayer& getLayer(unsigned int);
    unsigned int getLayerNumber();
    
    double getBiasSumGradient(unsigned int index) const;
    double getWeightSumGradient(unsigned int index) const;


    unsigned int _calcBiasIndex(const unsigned int layer, const unsigned int offset) const;
    
    void upgradeBias(double beta, double learnRate, bool rmsprop = true);
    void upgradeWeight(double beta, double learnRate, double regularization, bool rmsprop = true);

    void printMinMax();

    
    void serialize(std::ofstream& ss) const;
    bool deserialize(std::ifstream& ss);

    void setQuantization(bool q);
    
private:
    std::vector<DenseLayer> _parallelLayers;
    const unsigned int _number;
    const unsigned int _layerInputSize;
    const unsigned int _layerOutputSize;
    const unsigned int _layerWeightNumber;

};

#endif  
