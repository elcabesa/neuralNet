#include "gtest/gtest.h"
#include "dense.h" 

TEST(denseInputTest, constructor1) {
    std::vector<double> in = {3.0, 12, -75, 544, 913, 0, 32, 1e-7, 7};
    DenseInput input(in);
    
    ASSERT_EQ(input.size(), 9);
    ASSERT_EQ(input.getElementNumber(), 9);
    
    ASSERT_DOUBLE_EQ(input.get(0), 3.0);
    ASSERT_DOUBLE_EQ(input.get(1), 12);
    ASSERT_DOUBLE_EQ(input.get(2), -75);
    ASSERT_DOUBLE_EQ(input.get(3), 544);
    ASSERT_DOUBLE_EQ(input.get(4), 913);
    ASSERT_DOUBLE_EQ(input.get(5), 0);
    ASSERT_DOUBLE_EQ(input.get(6), 32);
    ASSERT_DOUBLE_EQ(input.get(7), 1e-7);
    ASSERT_DOUBLE_EQ(input.get(8), 7);
    
    ASSERT_DEATH(input.get(9), "");
    ASSERT_DEATH(input.get(300), "");
    
    ASSERT_EQ(input.getElementFromIndex(0).first, 0);
    ASSERT_EQ(input.getElementFromIndex(1).first, 1);
    ASSERT_EQ(input.getElementFromIndex(2).first, 2);
    ASSERT_EQ(input.getElementFromIndex(3).first, 3);
    ASSERT_EQ(input.getElementFromIndex(4).first, 4);
    ASSERT_EQ(input.getElementFromIndex(5).first, 5);
    ASSERT_EQ(input.getElementFromIndex(6).first, 6);
    ASSERT_EQ(input.getElementFromIndex(7).first, 7);
    ASSERT_EQ(input.getElementFromIndex(8).first, 8);
    
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(0).second, 3.0);
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(1).second, 12);
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(2).second, -75);
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(3).second, 544);
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(4).second, 913);
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(5).second, 0);
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(6).second, 32);
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(7).second, 1e-7);
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(8).second, 7);
    
    ASSERT_DEATH(input.getElementFromIndex(9), "");
    ASSERT_DEATH(input.getElementFromIndex(300), "");
    
    input.set(3,127.0);
    
    ASSERT_DOUBLE_EQ(input.get(0), 3.0);
    ASSERT_DOUBLE_EQ(input.get(1), 12);
    ASSERT_DOUBLE_EQ(input.get(2), -75);
    ASSERT_DOUBLE_EQ(input.get(3), 127.0);
    ASSERT_DOUBLE_EQ(input.get(4), 913);
    ASSERT_DOUBLE_EQ(input.get(5), 0);
    ASSERT_DOUBLE_EQ(input.get(6), 32);
    ASSERT_DOUBLE_EQ(input.get(7), 1e-7);
    ASSERT_DOUBLE_EQ(input.get(8), 7);
    
    ASSERT_DEATH(input.set(9,127.0), "");
 
}

TEST(denseInputTest, constructor2) {
    DenseInput input(127);
    
    ASSERT_EQ(input.size(), 127);
    ASSERT_EQ(input.getElementNumber(), 127);
    
    for(unsigned int i =0;i<127; ++i) {
        ASSERT_DOUBLE_EQ(input.get(i), 0.0);
    } 
    
    ASSERT_DEATH(input.get(127), "");
    ASSERT_DEATH(input.get(300), "");
    
     for(unsigned int i =0;i<127; ++i) {
         ASSERT_EQ(input.getElementFromIndex(i).first, i);
         ASSERT_DOUBLE_EQ(input.getElementFromIndex(i).second, 0.0);
    }
    
    ASSERT_DEATH(input.getElementFromIndex(127), "");
    ASSERT_DEATH(input.getElementFromIndex(300), "");
    
    input.set(3,127.0);
    
    for(unsigned int i =0;i<127; ++i) {
        if( i != 3) {
            ASSERT_DOUBLE_EQ(input.get(i), 0.0);
        }
        else {
            ASSERT_DOUBLE_EQ(input.get(i), 127.0);
        }
    } 
    ASSERT_DEATH(input.set(127,127.0), "");
 
}

TEST(denseInputTest, inputRef) {
    std::vector<double> in = {3.0, 12, -75, 544, 913, 0, 32, 1e-7, 7};
    DenseInput input(in);
    
    Input& ref = input;
    
    ASSERT_EQ(ref.size(), 9);
    ASSERT_EQ(ref.getElementNumber(), 9);
    
    ASSERT_DOUBLE_EQ(ref.get(0), 3.0);
    ASSERT_DOUBLE_EQ(ref.get(7), 1e-7);

    
    ASSERT_DEATH(ref.get(9), "");
    ASSERT_EQ(ref.getElementFromIndex(1).first, 1);

    ASSERT_DOUBLE_EQ(ref.getElementFromIndex(3).second, 544);

    ASSERT_DEATH(ref.getElementFromIndex(9), "");
    
    ref.set(3,127.0);

    ASSERT_DOUBLE_EQ(ref.get(3), 127.0);

    
    ASSERT_DEATH(ref.set(9,127.0), "");
 
}


