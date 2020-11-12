#ifndef _GRADIENT_DESCEND_H
#define _GRADIENT_DESCEND_H

class Model;
class InputSet;
class PedoneCheck;

class GradientDescend {
public:
    GradientDescend(Model& model, const InputSet& inputSet, unsigned int passes, double learnRate, double regularization, double beta, unsigned int quant, bool rmsprop);
    ~GradientDescend();
    
    double train(unsigned int decimation);
private:
    Model& _model;
    const InputSet& _inputSet;
    unsigned int _passes;
    double _learnRate;
    double _regularization;
    double _beta;
    
    void _pass(const unsigned int pass);
    void _printTrainResult(const unsigned int pass, unsigned int decimation);
    void _save(const unsigned int pass);
    
    double _min;
    double _accumulatorLoss;
    unsigned int _count;
    bool _quantization;
    unsigned int _quantizationPass;
    bool _rmsProp;
    double _totalLoss;
    unsigned long long int _totalCount;

    PedoneCheck* _pedone = nullptr;
    
};

#endif
