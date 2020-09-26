#include "gtest/gtest.h"
#include "parallelDenseLayer.h"
#include "linear.h"
#include "relu.h"

TEST(parallelDenseLayerTest, testPropagate1) {
    ParallelDenseLayer layer(2, 2, 1, ActivationFactory::create(ActivationFactory::type::linear));
    layer.propagate(DenseInput({0.0, 0.0, 0.0, 0.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 0.0);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 0.0);
    
    layer.propagate(DenseInput({2.0, 5.0, -2.0,  5.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 7.0);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 3.0);
    
}

TEST(parallelDenseLayerTest, testPropagate2) {
    ParallelDenseLayer layer(2, 2, 1, ActivationFactory::create(ActivationFactory::type::linear));
    layer.getLayer(0).bias() = {3.0};
    layer.getLayer(0).weight() = {1.0, 1.0};
    
    layer.getLayer(1).bias() = {-2.1};
    layer.getLayer(1).weight() = {1.0, 1.0};

    layer.propagate(DenseInput({0.0, 0.0, 0.0, 0.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 3.0);

    layer.propagate(DenseInput({2.0, 5.0, 1.7, 1.4}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 10.0);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 1.0);

    
}

TEST(parallelDenseLayerTest, testPropagate3) {
    ParallelDenseLayer layer(2, 4, 2, ActivationFactory::create(ActivationFactory::type::linear));
    
    layer.getLayer(0).bias() = {3.0, 1.1};
    layer.getLayer(0).weight() = {0.5, 0.7, -1.0, 0.2, 1.1,-1.0, 3.0,-0.9,};
    
    layer.getLayer(1).bias() = {-2.0, 1.5};
    layer.getLayer(1).weight() = {0.6, 0.4, -1.2, 0.3, 1.0,-1.1, 3.1,-0.7};

    layer.propagate(DenseInput({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 3.0);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 1.1);
    ASSERT_DOUBLE_EQ(layer.output().get(2), -2.0);
    ASSERT_DOUBLE_EQ(layer.output().get(3), 1.5);
    
    ASSERT_DOUBLE_EQ(layer.getOutput(0), 3.0);
    ASSERT_DOUBLE_EQ(layer.getOutput(1), 1.1);
    ASSERT_DOUBLE_EQ(layer.getOutput(2), -2.0);
    ASSERT_DOUBLE_EQ(layer.getOutput(3), 1.5);
    
    layer.propagate(DenseInput({2.0, 5.0, 1.0, 1.0, 3.0,-2.0, 4.0, 1.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 3.1);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 1.6);
    ASSERT_DOUBLE_EQ(layer.output().get(2), 9.3);
    ASSERT_DOUBLE_EQ(layer.output().get(3), -3.0);

    layer.propagate(DenseInput({2.0, -5.0, 1.0, 1.0, -3.0,-2.0, 4.0, 1.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 13.1);
    ASSERT_DOUBLE_EQ(layer.output().get(1), -0.4);
    ASSERT_DOUBLE_EQ(layer.output().get(2), 5.7);
    ASSERT_DOUBLE_EQ(layer.output().get(3), -5.4);
    
}

TEST(parallelDenseLayerTest, testPropagateRelu) {
    ParallelDenseLayer layer(2, 4, 2, ActivationFactory::create(ActivationFactory::type::relu));
    
    layer.getLayer(0).bias() = {3.0, 1.1};
    layer.getLayer(0).weight() = {0.5, 0.7, -1.0, 0.2, 1.1,-1.0, 3.0,-0.9,};
    
    layer.getLayer(1).bias() = {-2.0, 1.5};
    layer.getLayer(1).weight() = {0.6, 0.4, -1.2, 0.3, 1.0,-1.1, 3.1,-0.7};
    
    layer.propagate(DenseInput({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 3.0);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 1.1);
    ASSERT_DOUBLE_EQ(layer.output().get(2), -2e-5);
    ASSERT_DOUBLE_EQ(layer.output().get(3), 1.5);
    
    ASSERT_DOUBLE_EQ(layer.getOutput(0), 3.0);
    ASSERT_DOUBLE_EQ(layer.getOutput(1), 1.1);
    ASSERT_DOUBLE_EQ(layer.getOutput(2), -2e-5);
    ASSERT_DOUBLE_EQ(layer.getOutput(3), 1.5);
    
    layer.propagate(DenseInput({2.0, 5.0, 1.0, 1.0, 3.0,-2.0, 4.0, 1.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 3.1);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 1.6);
    ASSERT_DOUBLE_EQ(layer.output().get(2), 9.3);
    ASSERT_DOUBLE_EQ(layer.output().get(3), -3.0e-5);

    layer.propagate(DenseInput({2.0, -5.0, 1.0, 1.0, -3.0,-2.0, 4.0, 1.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 13.1);
    ASSERT_DOUBLE_EQ(layer.output().get(1), -0.4e-5);
    ASSERT_DOUBLE_EQ(layer.output().get(2), 5.7);
    ASSERT_DOUBLE_EQ(layer.output().get(3), -5.4e-5);
    
}

TEST(parallelDenseLayerTest, testGetInputSize) {
    ASSERT_EQ(ParallelDenseLayer(2, 4, 2, ActivationFactory::create(ActivationFactory::type::relu)).getInputSize(), 8);
    ASSERT_EQ(ParallelDenseLayer(4, 120, 2, ActivationFactory::create(ActivationFactory::type::linear)).getInputSize(), 480);
    ASSERT_EQ(ParallelDenseLayer(3, 7, 2, ActivationFactory::create(ActivationFactory::type::linear)).getInputSize(), 21);
    ASSERT_EQ(ParallelDenseLayer(5, 1, 2, ActivationFactory::create(ActivationFactory::type::relu)).getInputSize(), 5);
}

TEST(parallelDenseLayerTest, testGetOutputSize) {
    ASSERT_EQ(ParallelDenseLayer(2, 4, 2, ActivationFactory::create(ActivationFactory::type::relu)).getOutputSize(), 4);
    ASSERT_EQ(ParallelDenseLayer(4, 120, 7, ActivationFactory::create(ActivationFactory::type::linear)).getOutputSize(), 28);
    ASSERT_EQ(ParallelDenseLayer(3, 7, 12, ActivationFactory::create(ActivationFactory::type::linear)).getOutputSize(), 36);
    ASSERT_EQ(ParallelDenseLayer(5, 2, 32, ActivationFactory::create(ActivationFactory::type::relu)).getOutputSize(), 160);
}



