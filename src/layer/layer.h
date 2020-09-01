#ifndef _LAYER_H
#define _LAYER_H


class Layer {
public:
    Layer(const unsigned int inputSize, const unsigned int outputSize);
    virtual ~Layer();
    
    unsigned int getInputSize() const;
    unsigned int getOutputSize() const;
    double getOutput(unsigned int i) const;
    const std::vector<double>& output() const;
    void printOutput() const;
    
    virtual void propagate(const std::vector<double>& input) = 0;
    virtual void printParams() const = 0;
    virtual void randomizeParams() = 0;
    virtual void backwardCalcBias(const std::vector<double>& h) = 0;
    virtual void backwardCalcWeight(const std::vector<double>& prevOut) = 0;
    virtual std::vector<double> backPropHelper() const = 0;
    
    virtual void resetSum() = 0;
    virtual void accumulateGradients() = 0;
    
    virtual std::vector<double>& bias() = 0;
    virtual std::vector<double>& weight() = 0;
    
    virtual std::vector<double>& biasSumGradient() = 0;
    virtual std::vector<double>& weightSumGradient() = 0;

protected:
    unsigned int _inputSize;
    unsigned int _outputSize;
    std::vector<double> _output;
    
};

#endif  
