#include "gtest/gtest.h"
#include "relu.h" 

TEST(reluTest, testTransfer) {
	
	reluActivation relu;
	
	ASSERT_EQ(relu.propagate(-2), 0);
    ASSERT_EQ(relu.propagate(0), 0);
    ASSERT_EQ(relu.propagate(5), 5);
	
}
