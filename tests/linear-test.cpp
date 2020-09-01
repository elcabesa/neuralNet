#include "gtest/gtest.h"
#include "linear.h" 

TEST(linearTest, testTransfer) {
	
	linearActivation lin;
	
	ASSERT_EQ(lin.propagate(-2), -2);
    ASSERT_EQ(lin.propagate(0), 0);
    ASSERT_EQ(lin.propagate(5), 5);
	
}
