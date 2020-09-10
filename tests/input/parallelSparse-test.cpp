#include "gtest/gtest.h"
#include "sparse.h"
#include "parallelSparse.h" 


TEST(parallelSparseInputTest, constructorFail) {
    std::vector<unsigned int> in = {5,27,4,92,190};
    SparseInput input(200, in);
    
    ParalledSparseInput ps1(input, 0, 50);
    ParalledSparseInput ps2(input, 1, 50);
    ParalledSparseInput ps3(input, 2, 50);
    ParalledSparseInput ps4(input, 3, 50);
    ASSERT_DEATH(ParalledSparseInput ps5(input, 4, 50),"");
    
    ASSERT_DEATH(ParalledSparseInput ps5(input, 0, 51),"");
}

TEST(parallelSparseInputTest, parallel) {
    std::vector<unsigned int> in = {5,27,4,92,190};
    SparseInput input(200, in);
    
    ParalledSparseInput ps1(input, 0, 50);
    ParalledSparseInput ps2(input, 1, 50);
    ParalledSparseInput ps3(input, 2, 50);
    ParalledSparseInput ps4(input, 3, 50);
    
    ASSERT_EQ(ps1.size(), 50);
    ASSERT_EQ(ps2.size(), 50);
    ASSERT_EQ(ps3.size(), 50);
    ASSERT_EQ(ps4.size(), 50);
    
    for( unsigned int i = 0;i<50; ++i) {
        if(i == 4) {ASSERT_DOUBLE_EQ(ps1.get(i), 1.0);}
        else if(i == 5) {ASSERT_DOUBLE_EQ(ps1.get(i), 1.0);}
        else if(i == 27) {ASSERT_DOUBLE_EQ(ps1.get(i), 1.0);}
        else {ASSERT_DOUBLE_EQ(ps1.get(i), 0.0);}
        
        if(i == 42) {ASSERT_DOUBLE_EQ(ps2.get(i), 1.0);}
        else {ASSERT_DOUBLE_EQ(ps2.get(i), 0.0);}
        
        ASSERT_DOUBLE_EQ(ps3.get(i), 0.0);
        
        if(i == 40) {ASSERT_DOUBLE_EQ(ps4.get(i), 1.0);}
        else {ASSERT_DOUBLE_EQ(ps4.get(i), 0.0);}
    }
    
    ASSERT_DEATH(ps1.get(50), "");
    ASSERT_DEATH(ps2.get(50), "");
    ASSERT_DEATH(ps3.get(50), "");
    ASSERT_DEATH(ps4.get(50), "");
    
    // SET is not allowed
    ASSERT_DEATH(ps1.set(0, 12), "");
    ASSERT_DEATH(ps2.set(32, 7), "");
    ASSERT_DEATH(ps3.set(4, 5), "");
    ASSERT_DEATH(ps4.set(12, 6), "");
    
    ASSERT_EQ(ps1.getElementNumber(), 3);
    ASSERT_EQ(ps2.getElementNumber(), 1);
    ASSERT_EQ(ps3.getElementNumber(), 0);
    ASSERT_EQ(ps4.getElementNumber(), 1);
    
    ASSERT_EQ(ps1.getElementFromIndex(0).first, 4);
    ASSERT_EQ(ps1.getElementFromIndex(1).first, 5);
    ASSERT_EQ(ps1.getElementFromIndex(2).first, 27);
    
    ASSERT_EQ(ps1.getElementFromIndex(0).second, 1.0);
    ASSERT_EQ(ps1.getElementFromIndex(1).second, 1.0);
    ASSERT_EQ(ps1.getElementFromIndex(2).second, 1.0);
    ASSERT_DEATH(ps1.getElementFromIndex(3), "");
    
    ASSERT_EQ(ps2.getElementFromIndex(0).first, 42);
    ASSERT_EQ(ps2.getElementFromIndex(0).second, 1.0);
    ASSERT_DEATH(ps2.getElementFromIndex(1), "");
    
    ASSERT_DEATH(ps3.getElementFromIndex(0), "");
    
    ASSERT_EQ(ps4.getElementFromIndex(0).first, 40);
    ASSERT_EQ(ps4.getElementFromIndex(0).second, 1.0);
    ASSERT_DEATH(ps4.getElementFromIndex(1), "");
}

TEST(parallelSparseInputTest, clear) {
    std::vector<unsigned int> in = {5,27,4,92,190};
    SparseInput input(200, in);
    
    ParalledSparseInput ps1(input, 0, 50);
    
    ASSERT_DEATH(ps1.clear(), "");
    
}
