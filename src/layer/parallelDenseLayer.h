#ifndef _PARALLEL_DENSE_LAYER_H
#define _PARALLEL_DENSE_LAYER_H

#include <memory>
#include <vector>

#include "layer.h"
#include "denseLayer.h"

class Activation;

class ParallelDenseLayer: public Layer {
public:
    ParallelDenseLayer(const unsigned int number, const unsigned int inputSize, const unsigned int outputSize, std::shared_ptr<Activation> act, const unsigned int accumulatorBits, const double outScaling, const double stdDev = 0.0);
    ~ParallelDenseLayer();
    
    void propagate(const Input& input);
    void printParams() const;
    void randomizeParams();
    void backwardPropagate(const std::vector<double>& h, const Input& input);
    
    std::vector<double> backPropHelper() const;
    
    void resetSum();
    
    double getBiasSumGradient(unsigned int index) const;
    double getWeightSumGradient(unsigned int index) const;

    void upgradeBias(double beta, double learnRate, bool rmsprop = true);
    void upgradeWeight(double beta, double learnRate, double regularization, bool rmsprop = true);

    void printMinMax();
    
    void serialize(std::ofstream& ss) const;
    bool deserialize(std::ifstream& ss);
    
private:
    /*std::vector<DenseLayer> _parallelLayers;*/
    const unsigned int _number;
    const unsigned int _layerInputSize;
    const unsigned int _layerOutputSize;
    const unsigned int _layerWeightNumber;

    

    std::vector<double> _netOutput;
    std::shared_ptr<Activation> _act;

    std::set<unsigned int> _activeFeature;

    std::vector<double> _biasGradient;
    std::vector<double> _weightGradient;

    std::vector<double> _biasSumGradient;
    std::vector<double> _weightSumGradient;

    std::vector<double> _biasMovAvg;
    std::vector<double> _weightMovAvg;

    void _calcNetOut(const Input& input, unsigned int n);
    void _calcOut(unsigned int n);
    double _getQuantizedWeight(unsigned int) const;
    double _getQuantizedBias(unsigned int i) const;
    unsigned int _calcWeightIndex(const unsigned int i, const unsigned int o) const;
    void _backwardCalcBiasGradient(const std::vector<double>& h);
    void _backwardCalcWeightGradient(const Input& input);
    void _accumulateGradients(const Input& input);

    double _min = 1e8;
    double _max = -1e8;

};

#endif  
