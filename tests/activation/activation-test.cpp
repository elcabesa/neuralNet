#include "gtest/gtest.h"
#include "activation.h" 


TEST(ActivationFactory, createLinear) {
    
    auto ac = ActivationFactory::create(ActivationFactory::type::linear);
    
    ASSERT_EQ(ac->getType(), "Linear");
    
    ASSERT_DOUBLE_EQ(ac->propagate(-2000), -2000);
    ASSERT_DOUBLE_EQ(ac->propagate(0), 0);
    ASSERT_DOUBLE_EQ(ac->propagate(587), 587);
}

TEST(ActivationFactory, createRelu) {
    
    auto ac = ActivationFactory::create(ActivationFactory::type::relu);
    
    ASSERT_EQ(ac->getType(), "Relu");
    
    ASSERT_DOUBLE_EQ(ac->propagate(-2000), 0);
    ASSERT_DOUBLE_EQ(ac->propagate(0), 0);
    ASSERT_DOUBLE_EQ(ac->propagate(587), 587);
}

TEST(ActivationFactory, createWrong) {
    
    auto ac = ActivationFactory::create(ActivationFactory::type(12));
    
    ASSERT_EQ(ac->getType(), "Linear");
    
    ASSERT_DOUBLE_EQ(ac->propagate(-2000), -2000);
    ASSERT_DOUBLE_EQ(ac->propagate(0), 0);
    ASSERT_DOUBLE_EQ(ac->propagate(587), 587);
}

TEST(ActivationFactory, createDefault) {
    
    auto ac = ActivationFactory::create();
    
    ASSERT_EQ(ac->getType(), "Linear");
    
    ASSERT_DOUBLE_EQ(ac->propagate(-2000), -2000);
    ASSERT_DOUBLE_EQ(ac->propagate(0), 0);
    ASSERT_DOUBLE_EQ(ac->propagate(587), 587);
}
