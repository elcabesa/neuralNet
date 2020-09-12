#ifndef _GRADIENT_DESCEND_H
#define _GRADIENT_DESCEND_H

class Model;
class InputSet;

class GradientDescend {
public:
    GradientDescend(Model& model, const InputSet& inputSet, unsigned int passes, double learnRate, double regularization = 1.0);
    ~GradientDescend();
    
    double train();
private:
    Model& _model;
    const InputSet& _inputSet;
    unsigned int _passes;
    double _learnRate;
    double _regularization;
    
    void _pass();
    void _printTrainResult(const unsigned int pass);
    
    std::vector<std::vector<double>> _v_b;
    std::vector<std::vector<double>> _v_w;
    
    double _min;
    
};

#endif
