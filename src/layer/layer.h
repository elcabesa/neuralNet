#ifndef _LAYER_H
#define _LAYER_H

#include <fstream>
#include <iostream>
#include <memory>

#include "activation.h"
#include "dense.h"

class Layer {
public:
    Layer(const unsigned int inputSize, const unsigned int outputSize, const double stdDev, std::shared_ptr<Activation> act);
    virtual ~Layer();
    
    unsigned int getInputSize() const;
    unsigned int getOutputSize() const;
    double getOutput(unsigned int i) const;
    const Input& output() const;
    void printOutput() const;
    
    virtual void propagate(const Input& input) = 0;
    virtual void printParams() const = 0;
    virtual void randomizeParams() = 0;
    virtual void backwardPropagate(const std::vector<double>& h, const Input& input) = 0;
    virtual std::vector<double> backPropHelper() const = 0;
    
    virtual void resetSum() = 0;
    
    std::vector<double>& bias();
    std::vector<double>& weight();
    
    virtual double getBiasSumGradient(unsigned int index) const = 0;
    virtual double getWeightSumGradient(unsigned int index) const = 0;
    
    virtual void serialize(std::ofstream& ss) const = 0;
    virtual bool deserialize(std::ifstream& ss) = 0;
    
    virtual void upgradeBias(double beta, double learnRate, bool rmsprop = true) = 0;
    virtual void upgradeWeight(double beta, double learnRate, double regularization, bool rmsprop = true) = 0;

    virtual void printMinMax() = 0;

    void setQuantization(bool q);

    Activation::type getActivatinType() const;

protected:
    unsigned int _inputSize;
    unsigned int _outputSize;
    DenseInput _output;
    const double _stdDev;
    bool _quantization;

    std::shared_ptr<Activation> _act;

    std::vector<double> _bias;
    std::vector<double> _weight;
    
};

#endif  
