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
    double calcTotalLoss(const std::vector<std::shared_ptr<LabeledExample>>& input);
    void calcLossGradient(const LabeledExample& le);
    void calcTotalLossGradient(const std::vector<std::shared_ptr<LabeledExample>>& input);
    double train(unsigned int passes, double learnRate, const std::vector<std::shared_ptr<LabeledExample>>& trainSet, const std::vector<std::shared_ptr<LabeledExample>>& validationSet, double regularization = 0.99, const unsigned int decimation = 1);
    void pass(const std::vector<std::shared_ptr<LabeledExample>>& trainSet, double learnRate, double regularization);
    void printTrainResult(const unsigned int pass, const unsigned int passes, const unsigned int decimation, const std::vector<std::shared_ptr<LabeledExample>>& trainSet, const std::vector<std::shared_ptr<LabeledExample>>& validationSet);
    const std::vector<std::vector<std::shared_ptr<LabeledExample>>> createBatches( const std::vector<std::shared_ptr<LabeledExample>>& trainSet, const unsigned int batchSize);
    
    void serialize(std::ofstream& ss) const;
    bool deserialize(std::ifstream& ss);
    
private:
    std::vector<std::unique_ptr<Layer>> _layers;
    Cost cost;
    unsigned int decimationCount;
};

#endif
