#include "gtest/gtest.h"
#include "activation.h" 


TEST(ActivationFactory, createLinear) {
    
    auto ac = ActivationFactory::create(Activation::type::linear);
    
    ASSERT_EQ(ac->getType(), Activation::type::linear);
    
    ASSERT_DOUBLE_EQ(ac->propagate(-2000), -2000);
    ASSERT_DOUBLE_EQ(ac->propagate(0), 0);
    ASSERT_DOUBLE_EQ(ac->propagate(587), 587);
}

TEST(ActivationFactory, createRelu) {
    
    auto ac = ActivationFactory::create(Activation::type::relu);
    
    ASSERT_EQ(ac->getType(), Activation::type::relu);
    
    ASSERT_DOUBLE_EQ(ac->propagate(-2000), -2000 * 0.01);
    ASSERT_DOUBLE_EQ(ac->propagate(0), 0);
    ASSERT_DOUBLE_EQ(ac->propagate(587), 131.6);
}

TEST(ActivationFactory, createWrong) {
    
    auto ac = ActivationFactory::create(Activation::type(12));
    
    ASSERT_EQ(ac->getType(), Activation::type::linear);
    
    ASSERT_DOUBLE_EQ(ac->propagate(-2000), -2000);
    ASSERT_DOUBLE_EQ(ac->propagate(0), 0);
    ASSERT_DOUBLE_EQ(ac->propagate(587), 587);
}

TEST(ActivationFactory, createDefault) {
    
    auto ac = ActivationFactory::create();
    
    ASSERT_EQ(ac->getType(), Activation::type::linear);
    
    ASSERT_DOUBLE_EQ(ac->propagate(-2000), -2000);
    ASSERT_DOUBLE_EQ(ac->propagate(0), 0);
    ASSERT_DOUBLE_EQ(ac->propagate(587), 587);
}
