#include "gtest/gtest.h"
#include "denseLayer.h"
#include "linear.h"
#include "relu.h"

TEST(denseTest, testPropagate1) {
    DenseLayer layer(2, 1, ActivationFactory::create(Activation::type::linear), 16, 1.0);
    layer.propagate(DenseInput({0.0, 0.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 0.0);
    
    layer.propagate(DenseInput({2.0, 5.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 7.0);
    
    layer.propagate(DenseInput({-2.0, 5.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 3.0);
    
}

TEST(denseTest, testPropagate2) {
    DenseLayer layer(2, 1, ActivationFactory::create(Activation::type::linear), 16, 1.0);
    layer.bias() = {3.0};
    layer.weight() = {1.0, 1.0};

    layer.propagate(DenseInput({0.0, 0.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 3.0);

    layer.propagate(DenseInput({2.0, 5.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 10.0);

    layer.propagate(DenseInput({-2.0, 5.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 6.0);
    
}

TEST(denseTest, testPropagate3) {
    DenseLayer layer(4, 2, ActivationFactory::create(Activation::type::linear), 16, 1.0);
    
    layer.bias() = {3.0, 1.1};
    layer.weight() = {0.5, 0.7, -1.0, 0.2,
                      1.1,-1.0, 3.0,-0.9};

    layer.propagate(DenseInput({0.0, 0.0, 0.0, 0.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 3.0);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 1.1);
    
    ASSERT_DOUBLE_EQ(layer.getOutput(0), 3.0);
    ASSERT_DOUBLE_EQ(layer.getOutput(1), 1.1);
    
    layer.propagate(DenseInput({2.0, 5.0, 1.0, 1.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 3.1);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 1.6);

    layer.propagate(DenseInput({-2.0, 5.0, 7.0, 0.1}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 5.0);
    ASSERT_DOUBLE_EQ(layer.output().get(1), -6.39);
    
}

TEST(denseTest, testPropagateRelu) {
    DenseLayer layer(4, 2, ActivationFactory::create(Activation::type::relu), 16, 1.0);
    
    layer.bias() = {0.3, 0.11};
    layer.weight() = {0.05, 0.07, -0.1, 0.02,
                      0.11,-0.10, 0.3,-0.09};

    layer.propagate(DenseInput({0.0, 0.0, 0.0, 0.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 0.3);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 0.11);
    
    layer.propagate(DenseInput({2.0, 5.0, 1.0, 1.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 0.31);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 0.16);

    layer.propagate(DenseInput({-2.0, 5.0, 7.0, 0.1}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 0.50);
    ASSERT_DOUBLE_EQ(layer.output().get(1), -0.00639);
    
}

TEST(denseTest, testGetInputSize) {
    ASSERT_EQ(DenseLayer(4, 2, ActivationFactory::create(Activation::type::relu), 16, 1.0).getInputSize(), 4);
    ASSERT_EQ(DenseLayer(120, 2, ActivationFactory::create(Activation::type::linear), 16, 1.0).getInputSize(), 120);
    ASSERT_EQ(DenseLayer(7, 2, ActivationFactory::create(Activation::type::linear), 16, 1.0).getInputSize(), 7);
    ASSERT_EQ(DenseLayer(1, 2, ActivationFactory::create(Activation::type::relu), 16, 1.0).getInputSize(), 1);
}

TEST(denseTest, testGetOutputSize) {
    ASSERT_EQ(DenseLayer(4, 2, ActivationFactory::create(Activation::type::relu), 16, 1.0).getOutputSize(), 2);
    ASSERT_EQ(DenseLayer(120, 7, ActivationFactory::create(Activation::type::linear), 16, 1.0).getOutputSize(), 7);
    ASSERT_EQ(DenseLayer(7, 12, ActivationFactory::create(Activation::type::linear), 16, 1.0).getOutputSize(), 12);
    ASSERT_EQ(DenseLayer(1, 32, ActivationFactory::create(Activation::type::relu), 16, 1.0).getOutputSize(), 32);
}



