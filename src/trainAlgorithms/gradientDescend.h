#ifndef _GRADIENT_DESCEND_H
#define _GRADIENT_DESCEND_H

class Model;
class InputSet;

class GradientDescend {
public:
    GradientDescend(Model& model, const InputSet& inputSet, unsigned int passes, double learnRate, double regularization = 1.0,  double beta = 0.9);
    ~GradientDescend();
    
    double train();
private:
    Model& _model;
    const InputSet& _inputSet;
    unsigned int _passes;
    double _learnRate;
    double _regularization;
    double _beta;
    
    void _pass();
    void _printTrainResult(const unsigned int pass);
    
    double _min;
    
};

#endif
