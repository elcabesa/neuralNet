#ifndef _MODEL_H
#define _MODEL_h

#include <memory>
#include <vector>

#include "cost.h"
#include "layer.h"

class LabeledExample;
class Input;


class Model {
public:
    void addLayer(std::unique_ptr<Layer> l);
    void randomizeParams();
    void printParams();
    const Input& forwardPass(const Input& input, bool verbose = false);
    double calcLoss(const LabeledExample& le);
    double calcTotalLoss(const std::vector<LabeledExample>& input);
    void calcLossGradient(const LabeledExample& le);
    void calcTotalLossGradient(const std::vector<LabeledExample>& input);
    double train(unsigned int passes, double learnRate, const std::vector<LabeledExample>& trainSet, const std::vector<LabeledExample>& validationSet, double regularization = 0.99, const unsigned int decimation = 1);
    
private:
    std::vector<std::unique_ptr<Layer>> _layers;
    Cost cost;
};

#endif
