#include "gtest/gtest.h"
#include "relu.h" 


TEST(reluTest, propagate) {
    
    reluActivation relu;
    
    ASSERT_DOUBLE_EQ(relu.propagate(-2), -0.02);
    ASSERT_DOUBLE_EQ(relu.propagate(-1), -0.01);
    ASSERT_DOUBLE_EQ(relu.propagate(0), 0);
    ASSERT_DOUBLE_EQ(relu.propagate(0.5), 0.5);
    ASSERT_DOUBLE_EQ(relu.propagate(5), 5.0);
    ASSERT_DOUBLE_EQ(relu.propagate(127), 127.0);
    ASSERT_DOUBLE_EQ(relu.propagate(128), 127.01);
}

TEST(reluTest, derivate) {
    
    reluActivation relu;
    
    ASSERT_DOUBLE_EQ(relu.derivate(-2), 0.01);
    ASSERT_DOUBLE_EQ(relu.derivate(-1), 0.01);
    ASSERT_DOUBLE_EQ(relu.derivate(0.1), 1.0);
    ASSERT_DOUBLE_EQ(relu.derivate(0.5), 1.0);
    ASSERT_DOUBLE_EQ(relu.derivate(256), 0.01);
}

TEST(reluTest, getType) {
    
    reluActivation relu;
    
    ASSERT_EQ(relu.getType(), Activation::type::relu);
    
}
