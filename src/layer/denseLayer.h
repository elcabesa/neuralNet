#ifndef _DENSE_LAYER_H
#define _DENSE_LAYER_H

#include <cassert>
#include <memory>
#include <vector>
#include <set>

#include "layer.h"

class Activation;

class DenseLayer: public Layer {
public:
    DenseLayer(const unsigned int inputSize, const unsigned int outputSize, std::shared_ptr<Activation> act, unsigned int outScale, unsigned int weightScale, const double stdDev = 0.0);
    ~DenseLayer();
    
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
    
    void consolidateResult();
    
    double getBiasSumGradient(unsigned int index) const;
    double getWeightSumGradient(unsigned int index) const;
    
    void upgradeBias(double beta, double learnRate, bool rmsprop = true);
    void upgradeWeight(double beta, double learnRate, double regularization, bool rmsprop = true);
    
    unsigned int _calcWeightIndex(const unsigned int i, const unsigned int o) const;
    
    void serialize(std::ofstream& ss) const;
    bool deserialize(std::ifstream& ss);

    void printMinMax();
    void setQuantization(bool q);
    
private:
    std::vector<double> _bias;
    std::vector<double> _weight;
    std::vector<double> _quantizedWeight;

    std::vector<double> _netOutput;
    std::shared_ptr<Activation> _act;

    std::vector<double> _biasGradient;
    std::vector<double> _weightGradient;

    std::vector<double> _biasSumGradient;
    std::vector<double> _weightSumGradient;

    std::vector<double> _biasMovAvg;
    std::vector<double> _weightMovAvg;
    
    std::set<unsigned int> _activeFeature;
    
    void calcNetOut(const Input& input);
    void calcOut();

    double _min = 1e8;
    double _max = -1e8;

    void _quantizeWeight();
};

#endif  
