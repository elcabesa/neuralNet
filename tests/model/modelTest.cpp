#include "gtest/gtest.h"

#include "activation.h"
#include "labeledExample.h"
#include "model.h"
#include "parallelDenseLayer.h"
#include "denseLayer.h"

TEST(modelTest, layers) {
    Model m;
    m.addLayer(std::make_unique<ParallelDenseLayer>(2, 40960, 256, ActivationFactory::create(ActivationFactory::type::linear)));
    m.addLayer(std::make_unique<DenseLayer>(512,32, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(32,32, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(32, 1, ActivationFactory::create(ActivationFactory::type::linear)));
    
    ASSERT_EQ(m.getLayerCount(),4);
    
    ASSERT_EQ(m.getLayer(0).getInputSize(), 81920);
    ASSERT_EQ(m.getLayer(1).getInputSize(), 512);
    ASSERT_EQ(m.getLayer(2).getInputSize(), 32);
    ASSERT_EQ(m.getLayer(3).getInputSize(), 32);
    
    ASSERT_EQ(m.getLayer(0).getOutputSize(), 512);
    ASSERT_EQ(m.getLayer(1).getOutputSize(), 32);
    ASSERT_EQ(m.getLayer(2).getOutputSize(), 32);
    ASSERT_EQ(m.getLayer(3).getOutputSize(), 1);
    
    ASSERT_DEATH(m.getLayer(4),"");
    
}

TEST(modelTest, wrongSize) {
    Model m;
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(ActivationFactory::type::relu)));
    ASSERT_EXIT(m.addLayer(std::make_unique<DenseLayer>(23,1, ActivationFactory::create(ActivationFactory::type::linear))),::testing::ExitedWithCode(0),"");
}


TEST(modelTest, forwardPass) {
    Model m;
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(2,1, ActivationFactory::create(ActivationFactory::type::linear)));
    DenseInput in({2.5, 1.7});
    
    {
        auto& out = m.forwardPass(in);
        ASSERT_EQ(out.size(), 1);
        ASSERT_DOUBLE_EQ(out.get(0), 8.4);
    }
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.2};
    
    {
        auto& out = m.forwardPass(in);
        ASSERT_EQ(out.size(), 1);
        ASSERT_DOUBLE_EQ(out.get(0), 0.1492);
    }
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.3};
    
    {
        auto& out = m.forwardPass(in);
        ASSERT_EQ(out.size(), 1);
        ASSERT_DOUBLE_EQ(out.get(0), -0.2098);
    }
}

TEST(modelTest, forwardPassRelu) {
    Model m;
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(2,1, ActivationFactory::create(ActivationFactory::type::relu)));
    DenseInput in({2.5, 1.7});
    
    {
        auto& out = m.forwardPass(in);
        ASSERT_EQ(out.size(), 1);
        ASSERT_DOUBLE_EQ(out.get(0), 8.4);
    }
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.2};
    
    {
        auto& out = m.forwardPass(in);
        ASSERT_EQ(out.size(), 1);
        ASSERT_DOUBLE_EQ(out.get(0), 0.1492);
    }
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.3};
    
    {
        auto& out = m.forwardPass(in);
        ASSERT_EQ(out.size(), 1);
        ASSERT_DOUBLE_EQ(out.get(0), -0.2098e-5);
    }
}

TEST(modelTest, calcLoss) {
    Model m;
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(2,1, ActivationFactory::create(ActivationFactory::type::relu)));
    std::shared_ptr<Input> in(new DenseInput({2.5, 1.7}));
    LabeledExample le(std::move(in), 72000);
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.2};
    
    ASSERT_DOUBLE_EQ(m.calcLoss(le), 24.85689032);
}

TEST(modelTest, calcTotalLoss) {
    Model m;
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(ActivationFactory::type::relu)));
    m.addLayer(std::make_unique<DenseLayer>(2,1, ActivationFactory::create(ActivationFactory::type::relu)));
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.2};
    
    double localTotalLoss = 0.0;
    
    std::vector<std::shared_ptr<LabeledExample>> examples;
    {
        std::vector<double> inVec = {2.5, 1.7};
        std::shared_ptr<Input> in(new DenseInput(inVec));
        std::shared_ptr<LabeledExample> le(new LabeledExample(std::move(in),72000));
        localTotalLoss += m.calcLoss(*le);
        examples.push_back(std::move(le));
    }
    {
        std::vector<double> inVec = {1.0, -1.2};
        std::shared_ptr<Input> in(new DenseInput(inVec));
        std::shared_ptr<LabeledExample> le(new LabeledExample(std::move(in),-12000));
        localTotalLoss += m.calcLoss(*le);
        examples.push_back(std::move(le));
    }
    {
        std::vector<double> inVec = {0.4, 0.1};
        std::shared_ptr<Input> in(new DenseInput(inVec));
        std::shared_ptr<LabeledExample> le(new LabeledExample(std::move(in),40));
        localTotalLoss += m.calcLoss(*le);
        examples.push_back(std::move(le));
    }
    {
        std::vector<double> inVec = {1.5, 0.7};
        std::shared_ptr<Input> in(new DenseInput(inVec));
        std::shared_ptr<LabeledExample> le(new LabeledExample(std::move(in),250));
        localTotalLoss += m.calcLoss(*le);
        examples.push_back(std::move(le));
    }
    
    ASSERT_DOUBLE_EQ(m.calcAvgLoss(examples), localTotalLoss/4.0);
}

