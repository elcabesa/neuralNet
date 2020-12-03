#include "gtest/gtest.h"
#include "cost.h" 

TEST(costTest, calc) {
    Cost c;
    ASSERT_DOUBLE_EQ(c.calc(7.12, 4.46), 3.5378);
    ASSERT_DOUBLE_EQ(c.calc(4.46, 7.12), 3.5378);
    
    ASSERT_DOUBLE_EQ(c.calc(0, 0), 0);
    ASSERT_DOUBLE_EQ(c.calc(12, 12), 0);
    ASSERT_DOUBLE_EQ(c.calc(-12, -12), 0);
    
    ASSERT_DOUBLE_EQ(c.calc(500, -500), 500000);
}

TEST(costTest, derivate) {
    Cost c;
    ASSERT_DOUBLE_EQ(c.derivate(7.12, 4.46), 2.66);
    ASSERT_DOUBLE_EQ(c.derivate(4.46, 7.12), -2.66);
    
    ASSERT_DOUBLE_EQ(c.derivate(0, 0), 0);
    ASSERT_DOUBLE_EQ(c.derivate(12, 12), 0);
    ASSERT_DOUBLE_EQ(c.derivate(-12, -12), 0);
    
    ASSERT_DOUBLE_EQ(c.derivate(500, -500), 1000);
}

TEST(costTest, derivate2) {
    double epsilon = 1e-8;
    Cost c;
    double x1 = c.calc(526, 1122);
    double x2 = c.calc(526 + epsilon, 1122);

    ASSERT_NEAR(c.derivate(526,1122), (x2-x1)/epsilon, 1e-5);
}
