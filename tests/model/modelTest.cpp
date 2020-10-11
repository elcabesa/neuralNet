#include "gtest/gtest.h"

#include "activation.h"
#include "labeledExample.h"
#include "model.h"
#include "parallelDenseLayer.h"
#include "denseLayer.h"
#include "sparse.h"

TEST(modelTest, layers) {
    Model m;
    m.addLayer(std::make_unique<ParallelDenseLayer>(2, 40960, 256, ActivationFactory::create(Activation::type::linear), 1, 1));
    m.addLayer(std::make_unique<DenseLayer>(512,32, ActivationFactory::create(Activation::type::relu), 1, 1));
    m.addLayer(std::make_unique<DenseLayer>(32,32, ActivationFactory::create(Activation::type::relu), 1, 1));
    m.addLayer(std::make_unique<DenseLayer>(32, 1, ActivationFactory::create(Activation::type::linear), 1, 1));
    
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
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(Activation::type::relu), 1, 1));
    ASSERT_EXIT(m.addLayer(std::make_unique<DenseLayer>(23,1, ActivationFactory::create(Activation::type::linear), 1, 1)),::testing::ExitedWithCode(0),"");
}


TEST(modelTest, forwardPass) {
    Model m;
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(Activation::type::relu), 1, 1));
    m.addLayer(std::make_unique<DenseLayer>(2, 1, ActivationFactory::create(Activation::type::linear), 1, 1));
    DenseInput in({0.25, 0.17});
    
    {
        auto& out = m.forwardPass(in);
        ASSERT_EQ(out.size(), 1);
        ASSERT_DOUBLE_EQ(out.get(0), 0.84);
    }
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.2};
    
    {
        auto& out = m.forwardPass(in);
        ASSERT_EQ(out.size(), 1);
        ASSERT_DOUBLE_EQ(out.get(0), 0.70252);
    }
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.3};
    
    {
        auto& out = m.forwardPass(in);
        ASSERT_EQ(out.size(), 1);
        ASSERT_DOUBLE_EQ(out.get(0), 0.69362);
    }
}

TEST(modelTest, forwardPassRelu) {
    Model m;
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(Activation::type::relu), 1, 1));
    m.addLayer(std::make_unique<DenseLayer>(2,1, ActivationFactory::create(Activation::type::relu), 1, 1));
    DenseInput in({0.25, 0.17});
    
    {
        auto& out = m.forwardPass(in);
        ASSERT_EQ(out.size(), 1);
        ASSERT_DOUBLE_EQ(out.get(0), 0.84);
    }
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.2};
    
    {
        auto& out = m.forwardPass(in);
        ASSERT_EQ(out.size(), 1);
        ASSERT_DOUBLE_EQ(out.get(0), 0.70252);
    }
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.3};
    
    {
        auto& out = m.forwardPass(in);
        ASSERT_EQ(out.size(), 1);
        ASSERT_DOUBLE_EQ(out.get(0), 0.69362);
    }
}

TEST(modelTest, calcLoss) {
    Model m;
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(Activation::type::relu), 1, 1));
    m.addLayer(std::make_unique<DenseLayer>(2,1, ActivationFactory::create(Activation::type::relu), 1, 1));
    std::shared_ptr<Input> in(new DenseInput({2.5, 1.7}));
    LabeledExample le(std::move(in), 72000);
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.2};
    
    ASSERT_DOUBLE_EQ(m.calcLoss(le), 21.3725912402);
}

TEST(modelTest, calcTotalLoss) {
    Model m;
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(Activation::type::relu), 1, 1));
    m.addLayer(std::make_unique<DenseLayer>(2,1, ActivationFactory::create(Activation::type::relu), 1, 1));
    
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

TEST(modelTest, calcLossGradient) {
    Model m;
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(Activation::type::linear), 1, 1));
    m.addLayer(std::make_unique<DenseLayer>(2, 1, ActivationFactory::create(Activation::type::linear), 1, 1));
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.2};
    
    std::vector<std::shared_ptr<LabeledExample>> examples;
    std::vector<double> inVec = {2.5, 1.7};
    std::shared_ptr<Input> in(new DenseInput(inVec));
    std::shared_ptr<LabeledExample> le(new LabeledExample(std::move(in),250));
    examples.push_back(std::move(le));
    
    m.calcTotalLossGradient(examples);
    
    for(unsigned int l = 0; l < m.getLayerCount(); ++l) {
        auto& actualLayer = m.getLayer(l);
        for(unsigned int i = 0; auto& b : actualLayer.bias()) {
            auto originalB = b;
            b = originalB + 0.01;
            auto lplus = m.calcLoss(*(examples[0]));
            b = originalB - 0.01;
            auto lminus = m.calcLoss(*(examples[0]));
            b = originalB;
            double grad = (lplus - lminus)/(0.02);
            ASSERT_NEAR(actualLayer.getBiasSumGradient(i), grad, 1e-5);
            ++i;
        }

        for(unsigned int i = 0; auto& w : actualLayer.weight()) {
            auto originalW = w;
            w = originalW + 0.01;
            auto lplus = m.calcLoss(*(examples[0]));
            w = originalW - 0.01;
            auto lminus = m.calcLoss(*(examples[0]));
            w = originalW;
            double grad = (lplus - lminus)/(0.02);
            ASSERT_NEAR(actualLayer.getWeightSumGradient(i), grad, 1e-5);
            ++i;
        }
    }
    
}

TEST(modelTest, calcLossGradientRelu) {
    Model m;
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(Activation::type::relu), 1, 1));
    m.addLayer(std::make_unique<DenseLayer>(2, 1, ActivationFactory::create(Activation::type::linear), 1, 1));
    
    m.getLayer(0).bias() = {0.2, -0.3};
    m.getLayer(1).bias() = {0.5};
    
    m.getLayer(0).weight() = {0.2, 0.4, -0.2, 1.7};
    m.getLayer(1).weight() = {1.02, -0.2};
    
    std::vector<std::shared_ptr<LabeledExample>> examples;
    {
        std::vector<double> inVec = {2.5, 1.7};
        std::shared_ptr<Input> in(new DenseInput(inVec));
        std::shared_ptr<LabeledExample> le(new LabeledExample(std::move(in),250));
        examples.push_back(std::move(le));
    }
    {
        std::vector<double> inVec = {3.2, -2.0};
        std::shared_ptr<Input> in(new DenseInput(inVec));
        std::shared_ptr<LabeledExample> le(new LabeledExample(std::move(in),100));
        examples.push_back(std::move(le));
    }
    
    m.calcTotalLossGradient(examples);
    
    for(unsigned int l = 0; l < m.getLayerCount(); ++l) {
        auto& actualLayer = m.getLayer(l);
        for(unsigned int i = 0; auto& b : actualLayer.bias()) {
            double grad = 0.0;
            for(auto& e :examples) {
                auto originalB = b;
                b = originalB + 0.01;
                auto lplus = m.calcLoss(*(e));
                b = originalB - 0.01;
                auto lminus = m.calcLoss(*(e));
                b = originalB;
                grad += (lplus - lminus)/(0.02);
            }
            ASSERT_NEAR(actualLayer.getBiasSumGradient(i), grad, 1e-5);
            ++i;
        }

        for(unsigned int i = 0; auto& w : actualLayer.weight()) {
            double grad = 0;
            for(auto& e :examples) {
                auto originalW = w;
                w = originalW + 0.01;
                auto lplus = m.calcLoss(*(e));
                w = originalW - 0.01;
                auto lminus = m.calcLoss(*(e));
                w = originalW;
                grad += (lplus - lminus)/(0.02);
            }
            ASSERT_NEAR(actualLayer.getWeightSumGradient(i), grad, 1e-5);
            ++i;
        }
    }  
}

TEST(modelTest, calcLossGradientComplex) {
    Model m;

    m.addLayer(std::make_unique<ParallelDenseLayer>(2, 2, 2, ActivationFactory::create(Activation::type::linear), 1, 1));
    m.addLayer(std::make_unique<DenseLayer>(4, 2, ActivationFactory::create(Activation::type::relu), 1, 1));
    m.addLayer(std::make_unique<DenseLayer>(2, 2, ActivationFactory::create(Activation::type::relu), 1, 1));
    m.addLayer(std::make_unique<DenseLayer>(2, 1, ActivationFactory::create(Activation::type::linear), 1, 1));
    
    m.randomizeParams();
    
    std::vector<std::shared_ptr<LabeledExample>> examples;
    
    for( unsigned int count = 0; count < 2; ++count) {
        examples.clear();
        if(count == 0) {
            {
                std::vector<unsigned int> inVec = {0, 3};
                std::shared_ptr<Input> in(new SparseInput(4, inVec));
                std::shared_ptr<LabeledExample> le(new LabeledExample(std::move(in),-20));
                examples.push_back(std::move(le));
            }
            {
                std::vector<unsigned int> inVec = {1,2};
                std::shared_ptr<Input> in(new SparseInput(4, inVec));
                std::shared_ptr<LabeledExample> le(new LabeledExample(std::move(in),10));
                examples.push_back(std::move(le));
            }
        } else {
            {
                std::vector<unsigned int> inVec = {0, 1};
                std::shared_ptr<Input> in(new SparseInput(4,inVec));
                std::shared_ptr<LabeledExample> le(new LabeledExample(std::move(in),-5));
                examples.push_back(std::move(le));
            }
            {
                std::vector<unsigned int> inVec = {0,3};
                std::shared_ptr<Input> in(new SparseInput(4,inVec));
                std::shared_ptr<LabeledExample> le(new LabeledExample(std::move(in),2));
                examples.push_back(std::move(le));
            }
        }
        
        m.calcTotalLossGradient(examples);
        
        const double delta = 0.00001;
        //std::cout<<"eccomi"<<std::endl;
        for(unsigned int l = 0; l < m.getLayerCount(); ++l) {
            auto& actualLayer = m.getLayer(l);
            //std::cout<<"layer "<<l<<std::endl;
            ParallelDenseLayer*  pdl = dynamic_cast<ParallelDenseLayer*>(&actualLayer);
            if(!pdl) {
                for(unsigned int i = 0; auto& b : actualLayer.bias()) {
                    //std::cout<<"\tbias "<<i<<std::endl;
                    double grad = 0.0;
                    for(auto& e :examples) {
                        auto originalB = b;
                        b = originalB + delta;
                        auto lplus = m.calcLoss(*(e));
                        b = originalB - delta;
                        auto lminus = m.calcLoss(*(e));
                        b = originalB;
                        grad += (lplus - lminus)/(2.0 * delta);
                    }
                    ASSERT_NEAR(actualLayer.getBiasSumGradient(i), grad, 1e-6);
                    ++i;
                }

                for(unsigned int i = 0; auto& w : actualLayer.weight()) {
                    //std::cout<<"\tweight "<<i<<std::endl;
                    double grad = 0;
                    for(auto& e :examples) {
                        auto originalW = w;
                        w = originalW + delta;
                        auto lplus = m.calcLoss(*(e));
                        w = originalW - delta;
                        auto lminus = m.calcLoss(*(e));
                        w = originalW;
                        grad += (lplus - lminus)/(2.0 * delta);
                    }
                    ASSERT_NEAR(actualLayer.getWeightSumGradient(i), grad, 1e-6);
                    ++i;
                }
            } else {
                for(unsigned int n = 0; n < pdl->getLayerNumber(); ++n) {
                    //std::cout<<"\tparallel layer "<<n<<std::endl;
                    auto & layer = pdl->getLayer(n);
                    for(unsigned int i = 0; auto& b : layer.bias()) {
                        //std::cout<<"\t\tbias "<<i<<std::endl;
                        double grad = 0.0;
                        for(auto& e :examples) {
                            auto originalB = b;
                            b = originalB + delta;
                            auto lplus = m.calcLoss(*(e));
                            b = originalB - delta;
                            auto lminus = m.calcLoss(*(e));
                            b = originalB;
                            grad += (lplus - lminus)/(2.0 * delta);
                        }
                        ASSERT_NEAR(layer.getBiasSumGradient(i), grad, 1e-6);
                        ++i;
                    }

                    for(unsigned int i = 0; auto& w : layer.weight()) {
                        //std::cout<<"\t\tweight "<<i<<std::endl;
                        double grad = 0;
                        for(auto& e :examples) {
                            auto originalW = w;
                            w = originalW + delta;
                            auto lplus = m.calcLoss(*(e));
                            w = originalW - delta;
                            auto lminus = m.calcLoss(*(e));
                            w = originalW;
                            grad += (lplus - lminus)/(2.0 * delta);
                        }
                        ASSERT_NEAR(layer.getWeightSumGradient(i), grad, 1e-6);
                        ++i;
                    }
                }
            }
        }  
    }
}
