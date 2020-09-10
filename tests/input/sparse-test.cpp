#include "gtest/gtest.h"
#include "sparse.h" 

TEST(sparseInputTest, constructor1Death) {
    std::map<unsigned int, double> in = {{50,12}, {12,0.7}, {70,-42.5}, {99,0.1}, {102,127.98}};

    ASSERT_DEATH(SparseInput input(100, in),"");
}

TEST(sparseInputTest, constructor1) {
    std::map<unsigned int, double> in = {{50,12}, {12,0.7}, {70,-42.5}, {99,0.1}, {102,127.98}};

    SparseInput input(150, in);
    
    ASSERT_EQ(input.size(), 150);
    ASSERT_EQ(input.getElementNumber(), 5);
    
    for( unsigned int i = 0;i<150; ++i) {
        if(i == 50) {ASSERT_DOUBLE_EQ(input.get(i), 12);}
        else if(i == 12) {ASSERT_DOUBLE_EQ(input.get(i), 0.7);}
        else if(i == 70) {ASSERT_DOUBLE_EQ(input.get(i), -42.5);}
        else if(i == 99) {ASSERT_DOUBLE_EQ(input.get(i), 0.1);}
        else if(i == 102) {ASSERT_DOUBLE_EQ(input.get(i), 127.98);}
        else {ASSERT_DOUBLE_EQ(input.get(i), 0);}
    }
    ASSERT_DEATH(input.get(150), "");
    ASSERT_DEATH(input.get(900), "");
    
    ASSERT_EQ(input.getElementFromIndex(0).first, 12);
    ASSERT_EQ(input.getElementFromIndex(1).first, 50);
    ASSERT_EQ(input.getElementFromIndex(2).first, 70);
    ASSERT_EQ(input.getElementFromIndex(3).first, 99);
    ASSERT_EQ(input.getElementFromIndex(4).first, 102);
    
    ASSERT_EQ(input.getElementFromIndex(0).second, 0.7);
    ASSERT_EQ(input.getElementFromIndex(1).second, 12);
    ASSERT_EQ(input.getElementFromIndex(2).second, -42.5);
    ASSERT_EQ(input.getElementFromIndex(3).second, 0.1);
    ASSERT_EQ(input.getElementFromIndex(4).second, 127.98);
    ASSERT_DEATH(input.getElementFromIndex(6),"");
    
    ASSERT_DEATH(input.getElementFromIndex(300), "");
    
    input.set(12,127.0);
    
    for( unsigned int i = 0;i<150; ++i) {
        if(i == 50) {ASSERT_DOUBLE_EQ(input.get(i), 12);}
        else if(i == 12) {ASSERT_DOUBLE_EQ(input.get(i), 127.0);}
        else if(i == 70) {ASSERT_DOUBLE_EQ(input.get(i), -42.5);}
        else if(i == 99) {ASSERT_DOUBLE_EQ(input.get(i), 0.1);}
        else if(i == 102) {ASSERT_DOUBLE_EQ(input.get(i), 127.98);}
        else {ASSERT_DOUBLE_EQ(input.get(i), 0);}
    }
    
    ASSERT_DEATH(input.set(13,127.0), "");
    ASSERT_DEATH(input.set(150,0), "");
    ASSERT_DEATH(input.set(900,0), "");

}

TEST(sparseInputTest, constructor2) {
    std::vector<double> in = {3.0, 12, -75, 544, 913, 0, 32, 1e-7, 7};
    SparseInput input(in);
    
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

TEST(sparseInputTest, constructor3Death) {
    std::vector<unsigned int> in = {5,27,4,92,190};

    ASSERT_DEATH(SparseInput input(100, in),"");
}

TEST(sparseInputTest, constructor3) {
    std::vector<unsigned int> in = {5,27,4,92,190};

    SparseInput input(200, in);
    
    ASSERT_EQ(input.size(), 200);
    ASSERT_EQ(input.getElementNumber(), 5);
    
    for( unsigned int i = 0;i<200; ++i) {
        if(i == 5) {ASSERT_DOUBLE_EQ(input.get(i), 1.0);}
        else if(i == 27) {ASSERT_DOUBLE_EQ(input.get(i), 1.0);}
        else if(i == 4) {ASSERT_DOUBLE_EQ(input.get(i), 1.0);}
        else if(i == 92) {ASSERT_DOUBLE_EQ(input.get(i), 1.0);}
        else if(i == 190) {ASSERT_DOUBLE_EQ(input.get(i), 1.0);}
        else {ASSERT_DOUBLE_EQ(input.get(i), 0.0);}
    }
    ASSERT_DEATH(input.get(200), "");
    ASSERT_DEATH(input.get(900), "");
    
    ASSERT_EQ(input.getElementFromIndex(0).first, 4);
    ASSERT_EQ(input.getElementFromIndex(1).first, 5);
    ASSERT_EQ(input.getElementFromIndex(2).first, 27);
    ASSERT_EQ(input.getElementFromIndex(3).first, 92);
    ASSERT_EQ(input.getElementFromIndex(4).first, 190);
    
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(0).second, 1.0);
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(1).second, 1.0);
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(2).second, 1.0);
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(3).second, 1.0);
    ASSERT_DOUBLE_EQ(input.getElementFromIndex(4).second, 1.0);
    ASSERT_DEATH(input.getElementFromIndex(6),"");
    
    ASSERT_DEATH(input.getElementFromIndex(300), "");
    
    input.set(27,127.0);
    
    for( unsigned int i = 0;i<200; ++i) {
        if(i == 5) {ASSERT_DOUBLE_EQ(input.get(i), 1.0);}
        else if(i == 27) {ASSERT_DOUBLE_EQ(input.get(i), 127);}
        else if(i == 4) {ASSERT_DOUBLE_EQ(input.get(i), 1.0);}
        else if(i == 92) {ASSERT_DOUBLE_EQ(input.get(i), 1.0);}
        else if(i == 190) {ASSERT_DOUBLE_EQ(input.get(i), 1.0);}
        else {ASSERT_DOUBLE_EQ(input.get(i), 0.0);}
    }
    
    ASSERT_DEATH(input.set(13,127.0), "");
    ASSERT_DEATH(input.set(200,0), "");
    ASSERT_DEATH(input.set(900,0), "");

}

TEST(sparseInputTest, constructor4) {

    SparseInput input(300);
    
    ASSERT_EQ(input.size(), 300);
    ASSERT_EQ(input.getElementNumber(), 0);
    
    for( unsigned int i = 0;i<300; ++i) {
        ASSERT_DOUBLE_EQ(input.get(i), 0.0);
    }
    ASSERT_DEATH(input.get(300), "");
    ASSERT_DEATH(input.get(900), "");
    
    ASSERT_DEATH(input.getElementFromIndex(0),"");
    ASSERT_DEATH(input.set(0, 127.0), "");
    ASSERT_DEATH(input.set(1, 0), "");
    ASSERT_DEATH(input.set(900,0), "");

}


TEST(sparseInputTest, clear) {
    std::map<unsigned int, double> in = {{50,12}, {12,0.7}, {70,-42.5}, {99,0.1}, {102,127.98}};

    SparseInput input(150, in);
    input.clear();
    
    ASSERT_EQ(input.size(), 150);
    ASSERT_EQ(input.getElementNumber(), 0);
    
    for(unsigned int i = 0; i < 150; ++i) {
        ASSERT_DOUBLE_EQ(input.get(i), 0.0);
    }
}
