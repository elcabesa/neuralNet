#ifndef _INPUT_SET_H
#define _INPUT_SET_H

#include <memory>
#include <vector>

class LabeledExample;

class InputSet {
public:
    InputSet();
    ~InputSet();
    
    void generate();
    
    const std::vector<std::shared_ptr<LabeledExample>>& trainSet() const;
    const std::vector<std::shared_ptr<LabeledExample>>& validationSet() const;
    const std::vector<std::shared_ptr<LabeledExample>>& batch()const;

    
private:
    // TODO remove
    double function(double x1, double x2, double x3, double x4);
    std::vector<std::shared_ptr<LabeledExample>> _trainSet;
    std::vector<std::shared_ptr<LabeledExample>> _verificationSet;
    std::vector<std::vector<std::shared_ptr<LabeledExample>>> _batches;
    mutable unsigned int _n =0;
};

#endif 
