#include "gtest/gtest.h"
#include "denseLayer.h"
#include "linear.h"
#include "relu.h"

TEST(denseTest, testPropagate1) {
    DenseLayer layer(2, 1, ActivationFactory::create(ActivationFactory::type::linear));
    layer.propagate({0.0, 0.0});
    ASSERT_DOUBLE_EQ(layer.output()[0], 0.0);
    
    layer.propagate({2.0, 5.0});
    ASSERT_DOUBLE_EQ(layer.output()[0], 7.0);
    
    layer.propagate({-2.0, 5.0});
    ASSERT_DOUBLE_EQ(layer.output()[0], 3.0);
    
}

TEST(denseTest, testPropagate2) {
    DenseLayer layer(2, 1, ActivationFactory::create(ActivationFactory::type::linear));
    layer.bias() = {3.0};
    layer.weight() = {1.0, 1.0};

    layer.propagate({0.0, 0.0});
    ASSERT_DOUBLE_EQ(layer.output()[0], 3.0);

    layer.propagate({2.0, 5.0});
    ASSERT_DOUBLE_EQ(layer.output()[0], 10.0);

    layer.propagate({-2.0, 5.0});
    ASSERT_DOUBLE_EQ(layer.output()[0], 6.0);
    
}

TEST(denseTest, testPropagate3) {
    DenseLayer layer(4, 2, ActivationFactory::create(ActivationFactory::type::linear));
    
    layer.bias() = {3.0, 1.1};
    layer.weight() = {0.5, -1.0, 1.1, 3.0,
                        0.7, 0.2, -1.0, -0.9};

    layer.propagate({0.0, 0.0, 0.0, 0.0});
    ASSERT_DOUBLE_EQ(layer.output()[0], 3.0);
    ASSERT_DOUBLE_EQ(layer.output()[1], 1.1);
    
    layer.propagate({2.0, 5.0, 1.0, 1.0});
    ASSERT_DOUBLE_EQ(layer.output()[0], 3.1);
    ASSERT_DOUBLE_EQ(layer.output()[1], 1.6);

    layer.propagate({0-2.0, 5.0, 7.0, 0.1});
    ASSERT_DOUBLE_EQ(layer.output()[0], 5.0);
    ASSERT_DOUBLE_EQ(layer.output()[1], -6.39);
    
}

TEST(denseTest, testPropagateRelu) {
    DenseLayer layer(4, 2, ActivationFactory::create(ActivationFactory::type::relu));
    
    layer.bias() = {3.0, 1.1};
    layer.weight() = {0.5, -1.0, 1.1, 3.0,
                        0.7, 0.2, -1.0, -0.9};

    layer.propagate({0.0, 0.0, 0.0, 0.0});
    ASSERT_DOUBLE_EQ(layer.output()[0], 3.0);
    ASSERT_DOUBLE_EQ(layer.output()[1], 1.1);
    
    layer.propagate({2.0, 5.0, 1.0, 1.0});
    ASSERT_DOUBLE_EQ(layer.output()[0], 3.1);
    ASSERT_DOUBLE_EQ(layer.output()[1], 1.6);

    layer.propagate({0-2.0, 5.0, 7.0, 0.1});
    ASSERT_DOUBLE_EQ(layer.output()[0], 5.0);
    ASSERT_DOUBLE_EQ(layer.output()[1], 0);
    
}

