#ifndef _LAYER_H
#define _LAYER_H

#include <iostream>
#include <fstream>
#include "dense.h"

class Layer {
public:
    Layer(const unsigned int inputSize, const unsigned int outputSize, const unsigned int accumulatorBits, const double outScaling, const double stdDev);
    virtual ~Layer();
    
    unsigned int getInputSize() const;
    unsigned int getOutputSize() const;
    double getOutput(unsigned int i) const;
    const Input& output() const;
    void printOutput() const;
    
    virtual void propagate(const Input& input) = 0;
    virtual void printParams() const = 0;
    virtual void randomizeParams() = 0;
    virtual void backwardCalcBiasGradient(const std::vector<double>& h) = 0;
    virtual void backwardCalcWeightGradient(const Input& input) = 0;
    virtual std::vector<double> backPropHelper() const = 0;
    
    virtual void resetSum() = 0;
    virtual void accumulateGradients(const Input& input) = 0;
    
    virtual std::vector<double>& bias() = 0;
    virtual std::vector<double>& weight() = 0;
    
    
    virtual double getBiasSumGradient(unsigned int index) const = 0;
    virtual double getWeightSumGradient(unsigned int index) const = 0;
    
    virtual void serialize(std::ofstream& ss) const = 0;
    virtual bool deserialize(std::ifstream& ss) = 0;
    
    virtual void upgradeBias(double beta, double learnRate, bool rmsprop = true) = 0;
    virtual void upgradeWeight(double beta, double learnRate, double regularization, bool rmsprop = true) = 0;

    virtual void printMinMax() = 0;

    virtual void setQuantization(bool q) = 0;

protected:
    unsigned int _inputSize;
    unsigned int _outputSize;
    DenseInput _output;
    const double _stdDev;
    bool _quantization;
    unsigned int _accumulatorBits;
    double _outScaling;
    
};

#endif  
