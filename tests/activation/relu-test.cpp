#include "gtest/gtest.h"
#include "relu.h" 


TEST(reluTest, propagate) {
    
    reluActivation relu;
    
    ASSERT_DOUBLE_EQ(relu.propagate(-2), -0.2);
    ASSERT_DOUBLE_EQ(relu.propagate(0), 0);
    ASSERT_DOUBLE_EQ(relu.propagate(5), 5);
}

TEST(reluTest, derivate) {
    
    reluActivation relu;
    
    ASSERT_DOUBLE_EQ(relu.derivate(-2), 0.1);
    ASSERT_DOUBLE_EQ(relu.derivate(0), 1);
    ASSERT_DOUBLE_EQ(relu.derivate(5), 1);
}

TEST(reluTest, getType) {
    
    reluActivation relu;
    
    ASSERT_EQ(relu.getType(), "Relu");
    
}
