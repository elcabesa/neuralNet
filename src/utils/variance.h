#ifndef _VARIANCE_H
#define _VARIANCE_H

#include <cstdint>

class VarianceCalculator {
public:
    VarianceCalculator(); 
    
    void addValue(double x);
    double getMean() const;
    double getVariance() const;
    double getstdDev() const;
    void reset();
    void print();
private:
    uint64_t _n = 0;
    double _ex = 0.0;
    double _ex2 = 0.0;
};

#endif  
