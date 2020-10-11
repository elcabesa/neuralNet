#include "gtest/gtest.h"
#include "linear.h" 

TEST(linearTest, propagate) {
    
    linearActivation lin;
    
    ASSERT_DOUBLE_EQ(lin.propagate(-2), -2);
    ASSERT_DOUBLE_EQ(lin.propagate(0), 0);
    ASSERT_DOUBLE_EQ(lin.propagate(5), 5);
}

TEST(linearTest, derivate) {
    
    linearActivation lin;
    
    ASSERT_DOUBLE_EQ(lin.derivate(-2), 1);
    ASSERT_DOUBLE_EQ(lin.derivate(0), 1);
    ASSERT_DOUBLE_EQ(lin.derivate(5), 1);
}

TEST(linearTest, getType) {
    
    linearActivation lin;
    
    ASSERT_EQ(lin.getType(), Activation::type::linear);
    
}
