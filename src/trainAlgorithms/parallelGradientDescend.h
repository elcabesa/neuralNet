#ifndef _PARALLEL_GRADIENT_DESCEND_H
#define _PARALLEL_GRADIENT_DESCEND_H

class Model;
class InputSet;

class ParallelGradientDescend {
public:
    ParallelGradientDescend(Model& model, const InputSet& inputSet, unsigned int passes, double learnRate, double regularization = 1.0, double beta = 0.9);
    ~ParallelGradientDescend();
    
    double train();
private:
    Model& _model;
    const InputSet& _inputSet;
    unsigned int _passes;
    double _learnRate;
    double _regularization;
    unsigned int _counter = 0;
    double _beta;
    
    void _pass();
    void _calcLossGradient();
    void _updateParams();
    
    
    void _printTrainResult(const unsigned int pass);
    void _save(const unsigned int pass);
    
    std::vector<std::vector<double>> _v_b;
    std::vector<std::vector<double>> _v_w;
    
    double _min;
    
};

#endif
