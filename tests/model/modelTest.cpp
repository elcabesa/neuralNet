#include "gtest/gtest.h"
#include "parallelDenseLayer.h"
#include "linear.h"
#include "relu.h"

TEST(modelTest, testPropagate1) {
    ParallelDenseLayer layer(2, 2, 1, ActivationFactory::create(ActivationFactory::type::linear));
    layer.propagate(DenseInput({0.0, 0.0, 0.0, 0.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 0.0);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 0.0);
    
    layer.propagate(DenseInput({2.0, 5.0, -2.0,  5.0}));
    ASSERT_DOUBLE_EQ(layer.output().get(0), 7.0);
    ASSERT_DOUBLE_EQ(layer.output().get(1), 3.0);
    
}
